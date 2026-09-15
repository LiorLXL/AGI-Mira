import * as api from './api.js';
import { WorkspaceRepository } from './storage.js';
import * as ui from './ui.js';

const TERMINAL = new Set(['completed', 'failed', 'interrupted', 'unknown']);
const ACTIVE = new Set(['sending', 'streaming', 'stopping']);
const DOCUMENT_TOOLS = ['write_document', 'ingest_document', 'doc_agent'];
const makeId = () => globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
const now = () => new Date().toISOString();
const delaySave = new WeakMap();

/** @typedef {{id:string,title:string,createdAt:string,updatedAt:string,messageIds:string[],draft:string,useRag:boolean,selectedTools:string[],migrated?:boolean}} Session */
/** @typedef {{id:string,sessionId:string,role:'user'|'assistant'|'system',content:string,format:'text'|'markdown',status:string,steps:Array,toolCalls:Array,sources:Array,artifacts:Array,createdAt:string}} Message */
/** @typedef {{clientRunId:string,serverRequestId?:string,sessionId:string,assistantMessageId:string,state:string,answer:string,options:{useRag:boolean,selectedTools:string[]}} Run */
/** @typedef {{id:string,title:string,source:string,latestVersion:number,latestVersionId:string,indexStatus:string}} Document */
/** @typedef {{id:string,filename:string,status:string,error?:string,documentId?:string,versionId?:string,docHash?:string}} UploadJob */

function appendAnswer(existing, incoming) {
  const next = String(incoming ?? '');
  if (!next) return existing;
  if (!existing || existing === next) return existing || next;
  if (next.startsWith(existing)) return next;
  if (existing.endsWith(next)) return existing;
  return existing + next;
}

function currentRoute(hash = globalThis.location?.hash || '#/chat') {
  const match = hash.match(/^#\/(chat|documents|settings)(?:\/([^/]+))?/);
  let id = null;
  try { id = match?.[2] ? decodeURIComponent(match[2]) : null; } catch (_) {}
  return { page: match?.[1] || 'chat', id };
}

export class AppController {
  constructor({ repository = new WorkspaceRepository(), transport = api } = {}) {
    this.repository = repository; this.transport = transport;
    this.workspace = repository.load(); this.runs = new Map(); this.listeners = new Set();
    this.documents = { items: this.workspace.documents || [], selected: null, loading: false, error: '' };
    this.tools = { items: [], loading: false, error: '' };
    this.uploads = this.workspace.uploads || [];
  }

  subscribe(listener) { this.listeners.add(listener); return () => this.listeners.delete(listener); }
  emit(event) { this.listeners.forEach(listener => { try { listener(event, this); } catch (_) {} }); }
  persist(immediate = true) {
    this.workspace.uploads = this.uploads; this.workspace.documents = this.documents.items;
    if (immediate) { clearTimeout(delaySave.get(this)); delaySave.delete(this); this.repository.save(this.workspace); }
    else if (!delaySave.has(this)) delaySave.set(this, setTimeout(() => { delaySave.delete(this); this.repository.save(this.workspace); }, 250));
  }
  session(id) { return this.workspace.sessions.find(session => session.id === id) || null; }
  messages(id) { return this.workspace.messages.filter(message => message.sessionId === id); }
  message(id) { return this.workspace.messages.find(message => message.id === id) || null; }

  newSession(title = '新对话') {
    const time = now(); const session = { id: makeId(), title, createdAt: time, updatedAt: time, messageIds: [], draft: '', useRag: false, selectedTools: [] };
    this.workspace.sessions.unshift(session); this.persist(); this.emit({ type: 'sessions' }); return session;
  }
  updateSession(id, patch) { const session = this.session(id); if (!session) return null; Object.assign(session, patch, { updatedAt: now() }); this.persist(); this.emit({ type: 'sessions', sessionId: id }); return session; }
  removeSession(id) {
    if (this.runs.has(id) || this.messages(id).some(message => message.status === 'unknown')) return false;
    const before = this.workspace.sessions.length; this.workspace.sessions = this.workspace.sessions.filter(session => session.id !== id);
    if (this.workspace.sessions.length === before) return false;
    this.workspace.messages = this.workspace.messages.filter(message => message.sessionId !== id); this.persist(); this.emit({ type: 'sessions' }); return true;
  }
  addMessage(value) {
    const message = { id: value.id || makeId(), sessionId: value.sessionId, role: value.role, content: String(value.content || ''), format: value.format || 'markdown', status: value.status || 'completed', steps: value.steps || [], toolCalls: value.toolCalls || [], sources: value.sources || [], artifacts: value.artifacts || [], context: value.context || null, promptTrace: value.promptTrace || [], createdAt: value.createdAt || now() };
    this.workspace.messages.push(message); const session = this.session(message.sessionId); if (session && !session.messageIds.includes(message.id)) session.messageIds.push(message.id); this.persist(); return message;
  }
  hasUnknown(id) { return this.messages(id).some(message => message.status === 'unknown'); }

  applyRunToMessage(run, immediate = false) {
    const message = this.message(run.assistantMessageId); if (!message) return;
    Object.assign(message, { content: run.answer, status: run.state, steps: run.steps, toolCalls: run.toolCalls, sources: run.sources, artifacts: run.artifacts, context: run.context, promptTrace: run.promptTrace, memory: run.memory || '' });
    if (TERMINAL.has(run.state)) message.completedAt = now();
    this.persist(immediate || TERMINAL.has(run.state));
  }

  transition(run, state) { if (!TERMINAL.has(run.state)) run.state = state; }
  consume(run, event) {
    if (this.runs.get(run.sessionId)?.clientRunId !== run.clientRunId || TERMINAL.has(run.state)) return;
    const data = event.data || {};
    if (event.event === 'start') {
      if (data.session_id && data.session_id !== run.sessionId) throw new api.ProtocolError('流式响应的 session_id 不匹配', data);
      this.transition(run, 'streaming');
    } else if (event.event === 'route') { run.mode = data.mode || ''; this.transition(run, 'streaming'); }
    else if (event.event === 'token') { run.answer = appendAnswer(run.answer, data.content); this.transition(run, 'streaming'); }
    else if (event.event === 'step') { run.steps.push(data); if (DOCUMENT_TOOLS.some(name => String(data.tool || data.tool_name || '').includes(name))) run.refreshDocuments = true; this.transition(run, 'streaming'); }
    else if (event.event === 'tool_call') { run.toolCalls.push(data); if (DOCUMENT_TOOLS.some(name => String(data.tool_name || data.tool || '').includes(name))) run.refreshDocuments = true; this.transition(run, 'streaming'); }
    else if (event.event === 'rag_result') { run.sources.push(...(data.searchResults || [])); this.transition(run, 'streaming'); }
    else if (event.event === 'memory') { run.memory = data.extracted_info || ''; this.transition(run, 'streaming'); }
    else if (event.event === 'context') { run.context = data; this.transition(run, 'streaming'); }
    else if (event.event === 'done') {
      const response = data.response;
      if (response?.sessionId && response.sessionId !== run.sessionId) throw new api.ProtocolError('done 事件的 session_id 不匹配', response);
      run.answer = appendAnswer(run.answer, response?.answer); run.serverRequestId = response?.requestId || run.serverRequestId;
      if (response?.steps?.length) run.steps = response.steps;
      if (response?.toolCalls?.length) run.toolCalls = response.toolCalls;
      if (response?.toolCalls?.some(call => DOCUMENT_TOOLS.some(name => String(call.tool_name || call.tool || '').includes(name)))) run.refreshDocuments = true;
      if (response?.searchResults?.length) run.sources = response.searchResults;
      if (response?.artifacts?.length) { run.artifacts = response.artifacts; run.refreshDocuments = true; }
      run.promptTrace = response?.promptTrace || []; run.done = response;
      this.transition(run, response?.interrupted ? 'interrupted' : response?.success === false ? 'failed' : 'completed');
    } else if (event.event === 'unknown') run.diagnostics.push(event);
    this.applyRunToMessage(run, event.event !== 'token'); this.emit({ type: 'run', event, run });
  }

  async run(sessionId, input, options = {}) {
    const session = this.session(sessionId); const message = String(input || '').trim();
    if (!session) throw new Error('会话不存在');
    if (this.runs.has(sessionId)) { const error = new Error('这个会话已有任务在运行'); error.code = 'duplicate_run'; throw error; }
    if (this.hasUnknown(sessionId)) { const error = new Error('这个会话存在状态未知的请求，请新建会话继续'); error.code = 'unknown_run'; throw error; }
    if (!message) throw new Error('请输入消息');
    const user = this.addMessage({ sessionId, role: 'user', content: message, status: 'completed' });
    const assistant = this.addMessage({ sessionId, role: 'assistant', content: '', status: 'sending' });
    if (session.title === '新对话') session.title = message.slice(0, 24);
    session.draft = ''; session.updatedAt = now();
    const run = { clientRunId: makeId(), sessionId, userMessageId: user.id, assistantMessageId: assistant.id, options: { useRag: Boolean(options.useRag), selectedTools: [...(options.selectedTools || [])] }, state: 'sending', answer: '', mode: '', steps: [], toolCalls: [], sources: [], artifacts: [], promptTrace: [], context: null, diagnostics: [], refreshDocuments: false, streamSettled: false, cancelPending: false };
    run.streamSettledPromise = new Promise(resolve => { run.resolveStream = resolve; });
    this.runs.set(sessionId, run); this.persist(); this.emit({ type: 'run', event: { event: 'state' }, run });
    try {
      await this.transport.streamChat({ session_id: sessionId, message, use_rag: run.options.useRag, selected_tools: run.options.selectedTools, explicit: true }, event => this.consume(run, event), options.signal);
      if (!TERMINAL.has(run.state)) throw new api.StreamEOFError();
      return run;
    } catch (error) {
      if (!TERMINAL.has(run.state)) this.transition(run, error?.name === 'AbortError' || ['network_error', 'eof'].includes(error?.code) ? 'unknown' : 'failed');
      run.error = error; this.applyRunToMessage(run, true); this.emit({ type: 'run', event: { event: 'state', error }, run }); throw error;
    } finally {
      run.streamSettled = true; run.resolveStream(run);
      if (!run.cancelPending && this.runs.get(sessionId)?.clientRunId === run.clientRunId) this.runs.delete(sessionId);
      this.applyRunToMessage(run, true); this.emit({ type: 'run-settled', run });
      if (run.refreshDocuments) this.loadDocuments({ quiet: true });
    }
  }

  async stop(sessionId) {
    const run = this.runs.get(sessionId); if (!run || TERMINAL.has(run.state)) return { ok: false, skipped: true };
    if (run.cancelPending) return run.cancelPromise;
    this.transition(run, 'stopping'); run.cancelPending = true; this.applyRunToMessage(run, true); this.emit({ type: 'run', event: { event: 'state' }, run });
    run.cancelPromise = this.transport.cancel(sessionId);
    try { const result = await run.cancelPromise; run.cancelAccepted = result?.ok === true; if (run.cancelAccepted) await run.streamSettledPromise; return result; }
    catch (error) { run.cancelError = error; if (!TERMINAL.has(run.state)) this.transition(run, run.answer ? 'streaming' : 'sending'); this.applyRunToMessage(run, true); this.emit({ type: 'run', event: { event: 'cancel-error', error }, run }); throw error; }
    finally { run.cancelPending = false; if (run.streamSettled && this.runs.get(sessionId)?.clientRunId === run.clientRunId) this.runs.delete(sessionId); }
  }

  async loadDocuments({ quiet = false } = {}) {
    this.documents.loading = true; if (!quiet) this.emit({ type: 'documents' });
    try {
      const previous = new Map(this.documents.items.map(document => [document.id, document.indexStatus]));
      const items = await this.transport.listDocuments(); items.forEach(document => { document.indexStatus = previous.get(document.id) || 'unknown'; });
      const ids = new Set(items.map(document => document.id)); this.uploads.forEach(job => { job.linkedDocumentId = job.documentId && ids.has(job.documentId) ? job.documentId : undefined; });
      this.documents.items = items; this.documents.error = ''; this.workspace.documents = items; this.persist();
    } catch (error) { this.documents.error = error.message || '文档加载失败'; }
    finally { this.documents.loading = false; this.emit({ type: 'documents' }); }
  }
  async openDocument(id) {
    try { const detail = await this.transport.getDocument(id); const cached = this.documents.items.find(document => document.id === id); detail.document.indexStatus = cached?.indexStatus || 'unknown'; this.documents.selected = detail; this.documents.error = ''; this.emit({ type: 'documents' }); return detail; }
    catch (error) { this.documents.error = error.message || '读取文档失败'; this.emit({ type: 'documents' }); throw error; }
  }
  async ingestDocument(document, version) {
    const cached = this.documents.items.find(item => item.id === document.id); if (cached) cached.indexStatus = 'indexing'; if (this.documents.selected) this.documents.selected.document.indexStatus = 'indexing'; this.emit({ type: 'documents' });
    try {
      const result = await this.transport.ingestDocument(document.id, version?.id || document.latestVersionId);
      const indexed = Number(result.indexed_count || 0); const chunks = Number(result.chunk_count || 0); const state = indexed > 0 && indexed === chunks ? 'indexed' : 'failed';
      if (cached) cached.indexStatus = state; if (this.documents.selected) this.documents.selected.document.indexStatus = state;
      this.persist(); this.emit({ type: 'documents', message: state === 'indexed' ? `已入库 ${indexed}/${chunks} 个片段` : '入库未完成' }); return result;
    } catch (error) { if (cached) cached.indexStatus = 'failed'; if (this.documents.selected) this.documents.selected.document.indexStatus = 'failed'; this.persist(); this.emit({ type: 'documents', error }); throw error; }
  }
  async uploadFiles(files) {
    const jobs = [...files].map(file => ({ id: makeId(), filename: file.name, status: 'pending', createdAt: now(), file })); this.uploads.unshift(...jobs); this.emit({ type: 'uploads', jobs });
    for (const job of jobs) {
      job.status = 'uploading'; this.emit({ type: 'uploads', job });
      try {
        const result = await this.transport.uploadFile(job.file); job.result = result;
        if (result.needs_ocr) { job.status = 'needs_ocr'; job.error = result.message || '需要 OCR'; }
        else if (result.success === false) { job.status = 'failed'; job.error = result.message || '上传失败'; }
        else { job.status = 'completed'; job.documentId = result.document?.id || undefined; job.versionId = result.version?.id || undefined; job.docHash = result.doc_hash || undefined; }
      } catch (error) { job.status = 'failed'; job.error = error.message || '上传失败'; }
      delete job.file; this.persist(); this.emit({ type: 'uploads', job });
    }
    await this.loadDocuments({ quiet: true }); return jobs;
  }

  async loadTools() {
    this.tools.loading = true; this.emit({ type: 'tools' });
    try {
      const items = await this.transport.listTools(); const names = new Set(items.map(tool => tool.name)); const removed = [];
      this.workspace.sessions.forEach(session => { const valid = session.selectedTools.filter(name => names.has(name)); if (valid.length !== session.selectedTools.length) { removed.push(...session.selectedTools.filter(name => !names.has(name))); session.selectedTools = valid; } });
      this.tools.items = items; this.tools.error = ''; this.persist(); if (removed.length) this.repository.notice = `工具列表已变化，已移除不可用选择：${[...new Set(removed)].join('、')}`;
    } catch (error) { this.tools.error = error.message || '工具加载失败'; }
    finally { this.tools.loading = false; this.emit({ type: 'tools' }); }
  }
  setTool(sessionId, name, checked) { const session = this.session(sessionId); if (!session) return; const selected = new Set(session.selectedTools); checked ? selected.add(name) : selected.delete(name); session.selectedTools = [...selected]; this.persist(); this.emit({ type: 'sessions' }); }
  async registerTool(value) {
    const name = String(value.name || '').trim(); const endpoint = String(value.endpoint || '').trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) throw new Error('工具名称只能包含英文、数字和下划线，且不能以数字开头');
    let url; try { url = new URL(endpoint); } catch (_) { throw new Error('请输入有效的 HTTP 地址'); }
    if (!['http:', 'https:'].includes(url.protocol)) throw new Error('工具地址必须使用 http 或 https');
    const result = await this.transport.registerTool({ name, description: String(value.description || '').trim(), endpoint: url.href, params: [] }); await this.loadTools(); return result;
  }
}

function download(text, filename) {
  const url = URL.createObjectURL(new Blob([text], { type: 'application/json' })); const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function mount(controller, root = document) {
  ui.bindTheme(root, { initial: root.documentElement.dataset.themeMode || 'system', onChange: value => { try { localStorage.setItem('mira_theme', value); } catch (_) {} } });
  const mobileDrawer = ui.bindDrawer(root);
  const nodes = {
    pages: [...root.querySelectorAll('[data-page]')], title: root.querySelector('[data-page-title]'), notice: root.querySelector('[data-notice]'), service: root.querySelector('[data-service-status]'),
    sessions: root.querySelector('[data-session-list]'), mobileSessions: root.querySelector('[data-mobile-sessions]'), count: root.querySelector('[data-session-count]'), search: root.querySelector('[data-session-search]'),
    messages: root.querySelector('[data-message-list]'), messageScroll: root.querySelector('[data-message-scroll]'), welcome: root.querySelector('[data-welcome]'), composer: root.querySelector('[data-composer]'), input: root.querySelector('[data-composer-input]'), hint: root.querySelector('[data-composer-hint]'),
    rag: root.querySelector('[data-rag-toggle]'), selectedCount: root.querySelector('[data-selected-tool-count]'), send: root.querySelector('[data-action="send"]'), stop: root.querySelector('[data-action="stop"]'), more: root.querySelector('[data-action="session-menu"]'),
    docList: root.querySelector('[data-document-list]'), docReader: root.querySelector('[data-document-reader]'), docSearch: root.querySelector('[data-document-search]'), docSource: root.querySelector('[data-document-source]'), uploadJobs: root.querySelector('[data-upload-jobs]'),
    toolPicker: root.querySelector('[data-tool-picker]'), toolsError: root.querySelector('[data-tools-error]'), toolRegistry: root.querySelector('[data-tool-registry]'),
  };
  let route = currentRoute(); let activeId = null; let sessionQuery = ''; let documentQuery = ''; let documentSource = 'all'; let stickToBottom = true;
  const sessionForRoute = () => controller.session(activeId);
  const selectSession = id => { location.hash = `#/chat/${encodeURIComponent(id)}`; if (mobileDrawer?.open) mobileDrawer.close(); };
  const ensureSession = id => controller.session(id) || controller.workspace.sessions[0] || controller.newSession();
  const setNotice = (message, kind = 'info') => ui.showNotice(nodes.notice, message, kind);
  const renderSidebar = () => {
    const sessions = controller.workspace.sessions.filter(session => !sessionQuery || session.title.toLocaleLowerCase().includes(sessionQuery.toLocaleLowerCase()));
    ui.renderSessionLists(nodes.sessions, nodes.mobileSessions, sessions, activeId, controller.runs, { select: selectSession }); nodes.count.textContent = String(controller.workspace.sessions.length).padStart(2, '0');
    root.querySelectorAll('[data-route]').forEach(link => link.setAttribute('aria-current', link.dataset.route === route.page ? 'page' : 'false'));
  };
  const renderComposer = () => {
    const session = sessionForRoute(); const run = session && controller.runs.get(session.id);
    nodes.input.value = session?.draft || ''; nodes.rag.setAttribute('aria-pressed', String(Boolean(session?.useRag))); nodes.selectedCount.textContent = session?.selectedTools?.length ? `· ${session.selectedTools.length}` : '';
    nodes.send.hidden = Boolean(run); nodes.stop.hidden = !run; nodes.input.disabled = Boolean(run); nodes.hint.textContent = run ? ({ stopping: '正在发送停止信号…' }[run.state] || '回复生成中，可切换到其他会话') : 'Enter 发送，Shift+Enter 换行';
  };
  const renderChat = () => {
    const session = sessionForRoute(); const messages = session ? controller.messages(session.id) : [];
    nodes.welcome.hidden = messages.length > 0; ui.renderMessages(nodes.messages, messages); nodes.more.hidden = !session; nodes.title.textContent = session?.title || '新对话'; renderComposer();
    if (stickToBottom) nodes.messageScroll.scrollTop = nodes.messageScroll.scrollHeight;
  };
  const filteredDocuments = () => controller.documents.items.filter(document => (!documentQuery || document.title.toLocaleLowerCase().includes(documentQuery.toLocaleLowerCase())) && (documentSource === 'all' || document.source === documentSource));
  const renderDocuments = () => {
    ui.renderUploadJobs(nodes.uploadJobs, controller.uploads); ui.renderDocumentList(nodes.docList, filteredDocuments(), controller.documents.selected?.document?.id, id => { location.hash = `#/documents/${encodeURIComponent(id)}`; });
    ui.renderDocumentReader(nodes.docReader, controller.documents.selected, { ingest: (document, version) => controller.ingestDocument(document, version).catch(error => setNotice(error.message, 'error')) });
    if (controller.documents.error) setNotice(controller.documents.error, 'error');
  };
  const renderTools = () => {
    const session = sessionForRoute() || controller.workspace.sessions[0]; ui.renderToolPicker(nodes.toolPicker, controller.tools.items, session?.selectedTools || [], (name, checked) => { if (session) controller.setTool(session.id, name, checked); renderComposer(); }); ui.renderToolRegistry(nodes.toolRegistry, controller.tools.items);
    nodes.toolsError.hidden = !controller.tools.error; nodes.toolsError.textContent = controller.tools.error;
  };
  const renderRoute = () => {
    const nextRoute = currentRoute(); const pageChanged = nextRoute.page !== route.page; route = nextRoute;
    if (pageChanged && !controller.repository.notice) setNotice('');
    if (route.page === 'chat') { const session = ensureSession(route.id); activeId = session.id; if (route.id !== session.id) history.replaceState(null, '', `#/chat/${encodeURIComponent(session.id)}`); }
    nodes.pages.forEach(page => { page.hidden = page.dataset.page !== route.page; });
    nodes.title.textContent = route.page === 'chat' ? sessionForRoute()?.title || '新对话' : route.page === 'documents' ? '资料与成果' : '设置';
    if (route.page === 'documents') { renderDocuments(); if (route.id && route.id !== controller.documents.selected?.document?.id) controller.openDocument(route.id).catch(error => setNotice(error.message, 'error')); }
    if (route.page === 'settings') renderTools();
    renderSidebar(); if (route.page === 'chat') renderChat();
  };

  controller.subscribe(event => {
    if (controller.repository.notice) setNotice(controller.repository.notice, controller.repository.mode === 'protected' ? 'error' : 'info');
    if (event.type === 'run' && event.event?.event === 'token' && event.run.sessionId === activeId) { ui.scheduleMessageText(nodes.messages, event.run.assistantMessageId, event.run.answer); if (stickToBottom) requestAnimationFrame(() => { nodes.messageScroll.scrollTop = nodes.messageScroll.scrollHeight; }); renderSidebar(); return; }
    if (event.type.startsWith('document') || event.type === 'uploads') renderDocuments();
    if (event.type === 'uploads') {
      const uploading = controller.uploads.filter(job => ['pending', 'uploading'].includes(job.status)).length;
      const latest = event.job || controller.uploads[0];
      if (uploading) setNotice(`正在上传 ${uploading} 个文件…`);
      else if (latest?.status === 'needs_ocr') setNotice(`${latest.filename} 需要 OCR 后才能入库`, 'error');
      else if (latest?.status === 'failed') setNotice(`${latest.filename} 上传失败：${latest.error || '未知错误'}`, 'error');
      else if (latest) setNotice(`${latest.filename} 已上传，可在文档中查看。`);
    }
    if (event.type === 'tools') renderTools();
    renderSidebar(); if (route.page === 'chat') renderChat();
    if (event.message) setNotice(event.message); if (event.error) setNotice(event.error.message || '操作失败', 'error');
  });

  root.querySelectorAll('[data-action="new-session"]').forEach(button => button.addEventListener('click', () => selectSession(controller.newSession().id)));
  nodes.search.addEventListener('input', () => { sessionQuery = nodes.search.value.trim(); renderSidebar(); });
  root.querySelectorAll('[data-prompt]').forEach(button => button.addEventListener('click', () => { nodes.input.value = button.dataset.prompt; controller.updateSession(activeId, { draft: nodes.input.value }); nodes.input.focus(); }));
  nodes.input.addEventListener('input', () => controller.updateSession(activeId, { draft: nodes.input.value }));
  nodes.input.addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) { event.preventDefault(); nodes.composer.requestSubmit(); } });
  nodes.composer.addEventListener('submit', event => { event.preventDefault(); const session = sessionForRoute(); if (!session || !nodes.input.value.trim()) return; controller.run(session.id, nodes.input.value, { useRag: session.useRag, selectedTools: session.selectedTools }).catch(error => setNotice(error.message, 'error')); renderChat(); });
  nodes.rag.addEventListener('click', () => { const session = sessionForRoute(); if (session) controller.updateSession(session.id, { useRag: !session.useRag }); renderComposer(); });
  nodes.stop.addEventListener('click', () => { if (activeId) controller.stop(activeId).catch(error => setNotice(`停止请求失败：${error.message}`, 'error')); });
  nodes.more.addEventListener('click', () => { const dialog = root.querySelector('#sessionDialog'); const input = dialog.querySelector('[data-session-title]'); input.value = sessionForRoute()?.title || ''; dialog.showModal(); input.focus(); });
  root.querySelector('#sessionDialog [data-session-title]').addEventListener('change', event => { controller.updateSession(activeId, { title: event.target.value.trim() || '新对话' }); renderRoute(); });
  root.querySelector('#sessionDialog [data-action="remove-session"]').addEventListener('click', () => { if (!controller.removeSession(activeId)) { setNotice('运行中或状态未知的会话不能移除', 'error'); return; } root.querySelector('#sessionDialog').close(); const next = controller.workspace.sessions[0] || controller.newSession(); selectSession(next.id); });
  root.querySelector('[data-action="open-tools"]').addEventListener('click', () => { renderTools(); root.querySelector('#toolDialog').showModal(); });
  root.querySelector('[data-action="attach"]').addEventListener('click', () => root.querySelector('[data-chat-file]').click());
  root.querySelector('[data-action="upload-document"]').addEventListener('click', () => root.querySelector('[data-document-file]').click());
  const upload = event => { const files = [...event.target.files]; event.target.value = ''; if (!files.length) return; controller.uploadFiles(files); };
  root.querySelector('[data-chat-file]').addEventListener('change', upload); root.querySelector('[data-document-file]').addEventListener('change', upload);
  root.querySelector('[data-action="refresh-documents"]').addEventListener('click', () => controller.loadDocuments());
  nodes.docSearch.addEventListener('input', () => { documentQuery = nodes.docSearch.value.trim(); renderDocuments(); }); nodes.docSource.addEventListener('change', () => { documentSource = nodes.docSource.value; renderDocuments(); });
  root.querySelector('[data-tool-form]').addEventListener('submit', event => { event.preventDefault(); const formElement = event.currentTarget; const form = new FormData(formElement); controller.registerTool(Object.fromEntries(form)).then(() => { formElement.reset(); setNotice('工具已登记。当前为占位实现，尚未验证远程连接。'); }).catch(error => setNotice(error.message, 'error')); });
  const backupInput = root.querySelector('[data-backup-file]'); root.querySelector('[data-action="import-workspace"]').addEventListener('click', () => backupInput.click()); backupInput.addEventListener('change', async () => { const file = backupInput.files[0]; backupInput.value = ''; if (!file) return; try { controller.workspace = controller.repository.importBackup(await file.text(), controller.workspace); controller.uploads = controller.workspace.uploads; renderRoute(); setNotice('备份已导入；本地数据不会重建后端上下文。'); } catch (error) { setNotice(error.message, 'error'); } });
  root.querySelector('[data-action="export-workspace"]').addEventListener('click', () => download(controller.repository.exportWorkspace(controller.workspace), 'mira-workspace.json'));
  root.querySelector('[data-action="export-raw"]').addEventListener('click', () => { try { download(controller.repository.exportRaw(), 'mira-raw-backup.json'); } catch (error) { setNotice(error.message, 'error'); } });
  nodes.messageScroll.addEventListener('scroll', () => { stickToBottom = nodes.messageScroll.scrollHeight - nodes.messageScroll.scrollTop - nodes.messageScroll.clientHeight < 80; root.querySelector('[data-action="jump-latest"]').hidden = stickToBottom; });
  root.querySelector('[data-action="jump-latest"]').addEventListener('click', () => { stickToBottom = true; nodes.messageScroll.scrollTop = nodes.messageScroll.scrollHeight; });
  window.addEventListener('hashchange', renderRoute);
  controller.transport.getHealth().then(() => { nodes.service.dataset.state = 'online'; nodes.service.lastChild.textContent = '服务正常'; }).catch(() => { nodes.service.dataset.state = 'offline'; nodes.service.lastChild.textContent = '服务不可用'; });
  controller.loadTools(); controller.loadDocuments({ quiet: true });
  if (controller.repository.notice) setNotice(controller.repository.notice, controller.repository.mode === 'protected' ? 'error' : 'info');
  renderRoute(); return { renderRoute };
}

if (typeof document !== 'undefined') mount(new AppController(), document);
