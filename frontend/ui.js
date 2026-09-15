/** DOM rendering helpers. No fetch or localStorage access. */
export function make(tag, props = {}, children = []) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === 'text') element.textContent = value == null ? '' : String(value);
    else if (key === 'class') element.className = value;
    else if (key === 'dataset') Object.assign(element.dataset, value || {});
    else if (key === 'checked' || key === 'disabled' || key === 'hidden') element[key] = Boolean(value);
    else if (key.startsWith('on') && typeof value === 'function') element.addEventListener(key.slice(2), value);
    else if (value !== false && value != null) element.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of children) if (child != null) element.append(child.nodeType ? child : document.createTextNode(String(child)));
  return element;
}

function inline(text, parent) {
  const pattern = /(`[^`]*`|\*\*[^*]+\*\*|\*[^*]+\*|\[([^\]]+)\]\(([^\s)]+)\))/g;
  let cursor = 0;
  for (const match of String(text).matchAll(pattern)) {
    if (match.index > cursor) parent.append(document.createTextNode(text.slice(cursor, match.index)));
    const token = match[0];
    if (token.startsWith('`')) parent.append(make('code', { text: token.slice(1, -1) }));
    else if (token.startsWith('**')) parent.append(make('strong', { text: token.slice(2, -2) }));
    else if (token.startsWith('*')) parent.append(make('em', { text: token.slice(1, -1) }));
    else if (/^https?:\/\//i.test(match[3])) parent.append(make('a', { text: match[2], href: match[3], target: '_blank', rel: 'noopener noreferrer' }));
    else parent.append(document.createTextNode(token));
    cursor = match.index + token.length;
  }
  if (cursor < text.length) parent.append(document.createTextNode(text.slice(cursor)));
}

function codeBlock(content) {
  const wrapper = make('div', { class: 'code-block' });
  wrapper.append(make('pre', {}, [make('code', { text: content })]));
  const button = make('button', { type: 'button', class: 'code-copy', text: '复制代码' });
  button.addEventListener('click', async () => { try { await navigator.clipboard.writeText(content); button.textContent = '已复制'; } catch (_) { button.textContent = '复制失败'; } });
  wrapper.append(button);
  return wrapper;
}

function tableBlock(lines) {
  const rows = lines.map(line => line.replace(/^\||\|$/g, '').split('|').map(cell => cell.trim()));
  const wrapper = make('div', { class: 'markdown-table' });
  const table = make('table');
  const head = make('thead'); const body = make('tbody');
  const headerRow = make('tr'); rows[0].forEach(cell => headerRow.append(make('th', { text: cell }))); head.append(headerRow);
  rows.slice(2).forEach(values => { const row = make('tr'); values.forEach(cell => { const node = make('td'); inline(cell, node); row.append(node); }); body.append(row); });
  table.append(head, body); wrapper.append(table); return wrapper;
}

/** Safe, deliberately limited Markdown: headings, lists, quotes, links, tables and fenced code. */
export function renderMarkdown(text = '') {
  const root = make('div', { class: 'markdown-content' });
  const lines = String(text).replace(/\r\n?/g, '\n').split('\n');
  let index = 0; let list = null;
  const closeList = () => { if (list) { root.append(list); list = null; } };
  while (index < lines.length) {
    const line = lines[index];
    if (line.startsWith('```')) {
      closeList(); const collected = []; index += 1;
      while (index < lines.length && !lines[index].startsWith('```')) { collected.push(lines[index]); index += 1; }
      root.append(codeBlock(collected.join('\n'))); index += 1; continue;
    }
    if (line.includes('|') && index + 1 < lines.length && /^\s*\|?\s*:?-+/.test(lines[index + 1])) {
      closeList(); const collected = [line, lines[index + 1]]; index += 2;
      while (index < lines.length && lines[index].includes('|') && lines[index].trim()) { collected.push(lines[index]); index += 1; }
      root.append(tableBlock(collected)); continue;
    }
    if (!line.trim()) { closeList(); index += 1; continue; }
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    const item = line.match(/^\s*([-*+] |\d+[.)] )(.+)$/);
    if (heading) { closeList(); const element = make(`h${heading[1].length}`); inline(heading[2], element); root.append(element); index += 1; continue; }
    if (item) {
      const ordered = /^\d/.test(item[1]);
      if (!list || list.tagName !== (ordered ? 'OL' : 'UL')) { closeList(); list = make(ordered ? 'ol' : 'ul'); }
      const element = make('li'); inline(item[2], element); list.append(element); index += 1; continue;
    }
    closeList();
    if (line.startsWith('>')) { const quote = make('blockquote'); inline(line.replace(/^>\s?/, ''), quote); root.append(quote); }
    else { const paragraph = make('p'); inline(line, paragraph); root.append(paragraph); }
    index += 1;
  }
  closeList(); return root;
}

export function renderSessionLists(desktop, mobile, sessions, activeId, runs, actions) {
  const draw = root => {
    root.replaceChildren();
    for (const session of sessions) {
      const run = runs.get(session.id);
      const button = make('button', { type: 'button', class: `session-button${session.id === activeId ? ' active' : ''}`, dataset: { sessionId: session.id }, 'aria-current': session.id === activeId ? 'page' : 'false' });
      button.append(make('span', { text: run ? '●' : '○', class: run ? 'run-dot' : '' }), make('span', {}, [make('strong', { text: session.title || '新对话' }), make('small', { text: run ? ({ stopping: '正在停止…', unknown: '状态未知' }[run.state] || '运行中') : session.migrated ? '旧记录 · 本机' : '' })]));
      button.addEventListener('click', () => actions.select(session.id)); root.append(button);
    }
    if (!sessions.length) root.append(make('p', { class: 'session-empty', text: '还没有对话。' }));
  };
  draw(desktop); draw(mobile);
}

function statusLabel(status) {
  return { sending: '正在连接', streaming: '正在生成', stopping: '正在停止', completed: '', failed: '请求失败', interrupted: '已取消', unknown: '连接已中断，状态未知' }[status] ?? status;
}

export function renderMessage(root, message) {
  const article = make('article', { class: `message message-${message.role}`, dataset: { messageId: message.id, status: message.status } });
  article.append(make('header', { text: message.role === 'user' ? '你' : 'MIRA' }));
  const body = make('div', { class: 'message-body' }); body.append(renderMarkdown(message.content || (['sending', 'streaming', 'stopping'].includes(message.status) ? '…' : ''))); article.append(body);
  if (message.steps?.length || message.toolCalls?.length || message.context || message.promptTrace?.length) {
    const details = make('details', { class: 'execution-details' });
    details.append(make('summary', { text: `执行过程${message.steps?.length ? ` · ${message.steps.length} 步` : ''}` }));
    const content = make('div', { class: 'execution-body' });
    message.steps?.forEach(step => content.append(make('p', {}, [make('strong', { text: step.type || '步骤' }), `　${step.content || ''}`])));
    message.toolCalls?.forEach(call => content.append(make('div', { class: 'tool-record' }, [make('strong', { text: call.tool_name || '工具' }), make('p', { text: call.success === false ? `失败：${call.error || '未知错误'}` : String(call.tool_result || '') })])));
    if (message.memory) content.append(make('p', { class: 'trace-context', text: `本次记忆提取提示 · ${message.memory}` }));
    if (message.context) content.append(make('p', { class: 'trace-context', text: `Context · ${[message.context.mode, message.context.phase, message.context.prompt_version, Number.isFinite(message.context.prompt_chars) ? `${message.context.prompt_chars} chars` : ''].filter(Boolean).join(' · ')}` }));
    message.promptTrace?.forEach(trace => { const usage = [trace.mode, trace.phase, trace.prompt_version, Number.isFinite(trace.prompt_chars) ? `${trace.prompt_chars} chars` : '', Number.isFinite(trace.cached_tokens) ? `${trace.cached_tokens} cached tokens` : ''].filter(Boolean).join(' · '); content.append(make('p', { class: 'trace-context', text: `Prompt trace · ${usage || '可用'}` })); });
    details.append(content); article.append(details);
  }
  if (message.sources?.length) article.append(renderSources(message.sources));
  message.artifacts?.forEach(artifact => { if (artifact.documentId) article.append(make('a', { class: 'artifact-link', href: `#/documents/${encodeURIComponent(artifact.documentId)}`, text: `§ ${artifact.title || '查看生成文档'}` })); });
  const label = statusLabel(message.status); if (label) article.append(make('span', { class: 'message-status', dataset: { status: message.status }, text: label }));
  if (message.role === 'assistant' && message.content) { const copy = make('button', { type: 'button', class: 'copy-message', text: '复制回答' }); copy.addEventListener('click', () => navigator.clipboard?.writeText(message.content)); article.append(copy); }
  root.append(article); return article;
}

export function renderMessages(root, messages) { root.replaceChildren(); messages.forEach(message => renderMessage(root, message)); }

const messageFrames = new Map();
export function scheduleMessageText(root, messageId, text) {
  cancelAnimationFrame(messageFrames.get(messageId) || 0);
  messageFrames.set(messageId, requestAnimationFrame(() => {
    messageFrames.delete(messageId);
    const body = root.querySelector(`[data-message-id="${CSS.escape(messageId)}"] .message-body`);
    if (body) { body.replaceChildren(); body.append(renderMarkdown(text || '…')); }
  }));
}

export function renderSources(sources) {
  const container = make('div', { class: 'sources' });
  sources.forEach(source => {
    const content = [make('strong', { text: source.source || '来源' }), make('small', { text: source.content || '' })];
    if (source.url && /^https?:\/\//i.test(source.url)) container.append(make('a', { class: 'source-card', href: source.url, target: '_blank', rel: 'noopener noreferrer' }, content));
    else container.append(make('div', { class: 'source-card' }, content));
  });
  return container;
}

export function renderUploadJobs(root, jobs) {
  root.hidden = !jobs.length; root.replaceChildren();
  jobs.forEach(job => root.append(make('div', { class: 'upload-job', dataset: { status: job.status } }, [make('strong', { text: job.filename }), make('span', { text: job.error || ({ pending: '等待上传', uploading: '上传中', completed: job.documentId ? '已保存为文档' : '上传完成，未返回文档 ID', needs_ocr: '需要 OCR 后才能入库', failed: '上传失败', legacy: job.linkedDocumentId ? '已关联服务端文档 · 入库状态未知' : '旧缓存 · 入库状态未知' }[job.status] || job.status) })])));
}

export function renderDocumentList(root, documents, selectedId, onSelect) {
  root.replaceChildren();
  if (!documents.length) { root.append(make('div', { class: 'empty-state' }, [make('span', { text: '§' }), make('h2', { text: '暂无文档' }), make('p', { text: '上传资料，或让 Agent 生成一份成果。' })])); return; }
  documents.forEach(document => {
    const source = document.source === 'user_upload' ? '我的上传' : document.source === 'agent_generated' ? 'Agent 成果' : '其他来源';
    const button = make('button', { type: 'button', class: 'document-card', 'aria-current': document.id === selectedId ? 'true' : 'false' }, [make('strong', { text: document.title }), make('span', { text: `${source} · v${document.latestVersion || 0} · 索引状态${document.indexStatus === 'indexed' ? '已更新' : document.indexStatus === 'failed' ? '失败' : '未知'}` })]);
    button.addEventListener('click', () => onSelect(document.id)); root.append(button);
  });
}

export function renderDocumentReader(root, detail, actions = {}) {
  root.replaceChildren();
  if (!detail) { root.append(make('div', { class: 'empty-state' }, [make('span', { text: '§' }), make('h2', { text: '选择一份文档' }), make('p', { text: '阅读最新版本，或重新加入知识库。' })])); return; }
  const { document, version } = detail;
  const header = make('header');
  header.append(make('div', {}, [make('p', { class: 'eyebrow', text: 'LATEST VERSION' }), make('h2', { text: document.title }), make('p', { class: 'document-meta', text: `${document.source === 'user_upload' ? '我的上传' : 'Agent 成果'} · v${version.version || document.latestVersion || 0} · ${document.parser || version.metadata?.parser || '解析器未知'} · 索引状态${document.indexStatus === 'indexed' ? '已更新' : document.indexStatus === 'failed' ? '失败' : '未知'}` })]));
  const controls = make('div', { class: 'document-reader-actions' });
  const ingest = make('button', { type: 'button', text: document.indexStatus === 'indexing' ? '入库中…' : '重新入库', disabled: document.indexStatus === 'indexing' }); ingest.addEventListener('click', () => actions.ingest?.(document, version)); controls.append(ingest); header.append(controls); root.append(header);
  const content = make('div', { class: 'document-reader-content' }); content.append(renderMarkdown(version.content || '这份文档没有可显示的正文。')); root.append(content);
}

export function renderToolPicker(root, tools, selected, onChange) {
  root.replaceChildren();
  if (!tools.length) { root.append(make('p', { class: 'session-empty', text: '暂无可用工具。' })); return; }
  tools.forEach(tool => {
    const input = make('input', { type: 'checkbox', checked: selected.includes(tool.name), dataset: { tool: tool.name } }); input.addEventListener('change', () => onChange(tool.name, input.checked));
    root.append(make('label', { class: 'tool-option' }, [input, make('span', {}, [make('strong', { text: tool.name }), make('small', { text: tool.description || '无说明' })])]));
  });
}

export function renderToolRegistry(root, tools) {
  root.replaceChildren();
  tools.forEach(tool => root.append(make('div', { class: 'tool-item' }, [make('strong', { text: tool.name }), make('span', { text: tool.description || '无说明' }), make('small', { text: tool.isMcp ? '登记工具 · 占位实现' : '内置工具' })])));
}

export function bindTheme(root = document, options = {}) {
  const media = matchMedia('(prefers-color-scheme: dark)');
  const apply = mode => {
    const value = ['light', 'dark', 'system'].includes(mode) ? mode : 'system';
    root.documentElement.dataset.themeMode = value; root.documentElement.dataset.theme = value === 'system' ? (media.matches ? 'dark' : 'light') : value;
    root.querySelectorAll('[data-set-theme]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.setTheme === value)));
    options.onChange?.(value);
  };
  root.querySelectorAll('[data-set-theme]').forEach(button => button.addEventListener('click', () => apply(button.dataset.setTheme)));
  media.addEventListener('change', () => { if (root.documentElement.dataset.themeMode === 'system') apply('system'); });
  apply(options.initial || root.documentElement.dataset.themeMode || 'system'); return { apply };
}

export function bindDrawer(root = document) {
  const drawer = root.querySelector('#mobileDrawer'); const opener = root.querySelector('[data-open-drawer]'); const closer = root.querySelector('[data-close-drawer]');
  opener.addEventListener('click', () => drawer.showModal()); closer.addEventListener('click', () => drawer.close()); drawer.addEventListener('click', event => { if (event.target === drawer) drawer.close(); }); drawer.addEventListener('close', () => opener.focus());
  return drawer;
}

export function showNotice(root, message, kind = 'info') { root.textContent = message || ''; root.dataset.kind = kind; root.hidden = !message; }
export function announce(message) { const root = document.querySelector('#liveStatus'); if (root) root.textContent = message; }
