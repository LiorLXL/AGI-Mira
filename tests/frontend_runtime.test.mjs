import test from 'node:test';
import assert from 'node:assert/strict';
import { parseSSE, normalizeDocument, normalizeDocumentDetail, normalizeEvent, normalizeResponse } from '../frontend/api.js';
import { AppController } from '../frontend/app.js';
import { WorkspaceRepository, emptyWorkspace, STORAGE_KEY } from '../frontend/storage.js';

const encoder = new TextEncoder();
function streamFrom(parts) {
  return new ReadableStream({ start(controller) { parts.forEach(part => controller.enqueue(typeof part === 'string' ? encoder.encode(part) : part)); controller.close(); } });
}
async function collect(iterable) { const result = []; for await (const item of iterable) result.push(item); return result; }

test('SSE parser handles UTF-8 splits, CRLF, comments, multi-line data and terminal marker', async () => {
  const bytes = encoder.encode(': ping\r\nevent: token\r\ndata: {"content":"你好"}\r\n\r\nevent: future\ndata: {"x":1}\n\nevent: done\ndata: {"answer":\n' + 'data: "完成","interrupted":false}\n\ndata: [DONE]\n\n');
  const events = await collect(parseSSE(streamFrom([bytes.slice(0, 31), bytes.slice(31, 47), bytes.slice(47)])));
  assert.equal(events[0].event, 'token'); assert.equal(events[0].data.content, '你好');
  assert.deepEqual(events[1], { event: 'unknown', name: 'future', payload: { x: 1 } });
  assert.equal(events[2].event, 'done'); assert.equal(events[2].data.response.answer, '完成');
  assert.deepEqual(events[3], { event: 'transport_end', terminal: true });
});

test('wire adapters preserve current document, RAG and context shapes', () => {
  const document = normalizeDocument({ id: 'd1', title: '同名.md', source: 'user_upload', latest_version: 2, latest_version_id: 'v2', latest_metadata: { parser: 'plain_text' } });
  assert.deepEqual({ id: document.id, source: document.source, latestVersion: document.latestVersion, latestVersionId: document.latestVersionId }, { id: 'd1', source: 'user_upload', latestVersion: 2, latestVersionId: 'v2' });
  assert.equal(normalizeDocumentDetail({ document: { id: 'd1' }, version: { id: 'v2', content_md: '# 内容' } }).version.content, '# 内容');
  assert.equal(normalizeEvent('rag_result', { search_results: [{ chunk: { content: '片段' }, similarity: .7 }] }).data.searchResults[0].content, '片段');
  assert.deepEqual(normalizeEvent('context', { mode: 'chat', slots: [] }).data, { mode: 'chat', slots: [] });
});

class MemoryRepository {
  constructor(workspace = emptyWorkspace()) { this.value = structuredClone(workspace); this.mode = 'saved'; this.notice = ''; }
  load() { return structuredClone(this.value); }
  save(value) { this.value = structuredClone(value); return true; }
}

class FakeTransport {
  constructor() {
    this.streams = new Map(); this.cancelled = []; this.docs = [
      { id: 'd1', title: '同名.md', source: 'user_upload', latestVersion: 1, latestVersionId: 'v1', indexStatus: 'unknown' },
      { id: 'd2', title: '同名.md', source: 'agent_generated', latestVersion: 2, latestVersionId: 'v2', indexStatus: 'unknown' },
    ];
    this.toolRows = [{ name: 'search_web', description: '搜索', params: [], isMcp: false }];
  }
  streamChat(payload, onEvent) { return new Promise((resolve, reject) => this.streams.set(payload.session_id, { payload, onEvent, resolve, reject })); }
  event(sessionId, event) { this.streams.get(sessionId).onEvent(event); }
  finish(sessionId, response) { const stream = this.streams.get(sessionId); stream.onEvent({ event: 'done', data: { response } }); stream.resolve(response); }
  fail(sessionId, error) { this.streams.get(sessionId).reject(error); }
  cancel(sessionId) { this.cancelled.push(sessionId); return Promise.resolve({ ok: true }); }
  listDocuments() { this.documentLoads = (this.documentLoads || 0) + 1; if (this.failDocuments) return Promise.reject(new Error('文档服务暂不可用')); return Promise.resolve(structuredClone(this.docs)); }
  getDocument(id) { const document = this.docs.find(row => row.id === id); return Promise.resolve({ document: structuredClone(document), version: { id: document.latestVersionId, version: document.latestVersion, content: '# 正文', metadata: {} } }); }
  ingestDocument(id, versionId) { return Promise.resolve({ document_id: id, version_id: versionId, chunk_count: 2, indexed_count: 2 }); }
  uploadFile(file) { return Promise.resolve(file.name.startsWith('scan') ? { needs_ocr: true, message: '需要 OCR' } : { success: true, document: { id: `uploaded-${file.marker}` }, version: { id: `version-${file.marker}` }, doc_hash: `hash-${file.marker}` }); }
  listTools() { return Promise.resolve(structuredClone(this.toolRows)); }
  registerTool(value) { this.registered = value; this.toolRows.push({ name: value.name, description: value.description, params: [], isMcp: true }); return Promise.resolve({ success: true, ok: true }); }
  getHealth() { return Promise.resolve({ status: 'ok' }); }
}

test('controller pins concurrent streams to their originating sessions and blocks duplicates', async () => {
  const transport = new FakeTransport(); const app = new AppController({ repository: new MemoryRepository(), transport });
  const a = app.newSession('A'); const b = app.newSession('B');
  const first = app.run(a.id, '问题 A'); const second = app.run(b.id, '问题 B');
  await assert.rejects(() => app.run(a.id, '重复'), error => error.code === 'duplicate_run');
  transport.event(a.id, { event: 'token', data: { content: '回答 A' } }); transport.event(b.id, { event: 'token', data: { content: '回答 B' } });
  transport.finish(b.id, { answer: '回答 B', sessionId: b.id, interrupted: false }); transport.finish(a.id, { answer: '回答 A', sessionId: a.id, interrupted: false });
  await Promise.all([first, second]);
  assert.equal(app.messages(a.id).at(-1).content, '回答 A'); assert.equal(app.messages(b.id).at(-1).content, '回答 B');
  assert.equal(app.messages(a.id).at(-1).status, 'completed'); assert.equal(app.runs.size, 0);
});

test('controller consumes every supported event shape and preserves trace/tool/source details', async () => {
  const transport = new FakeTransport(); const app = new AppController({ repository: new MemoryRepository(), transport }); const session = app.newSession();
  const running = app.run(session.id, '复杂任务', { useRag: true, selectedTools: ['search_web'] });
  transport.event(session.id, { event: 'start', data: { session_id: session.id } });
  transport.event(session.id, { event: 'route', data: { mode: 'react' } });
  transport.event(session.id, { event: 'context', data: { mode: 'react', phase: 'generate', prompt_version: '001.1', prompt_chars: 12, slots: [] } });
  transport.event(session.id, { event: 'memory', data: { extracted_info: '偏好简洁回答' } });
  transport.event(session.id, { event: 'step', data: { type: 'Observation', content: '观察结果' } });
  transport.event(session.id, { event: 'tool_call', data: { tool_name: 'search_web', tool_result: '结果', success: true } });
  transport.event(session.id, { event: 'rag_result', data: { searchResults: [{ id: 'r1', content: '片段', source: '文档', score: .8 }] } });
  transport.event(session.id, { event: 'token', data: { content: '完成' } });
  transport.finish(session.id, { answer: '完成', sessionId: session.id, mode: 'react', promptTrace: [{ mode: 'react', prompt_chars: 20 }], interrupted: false });
  await running; const message = app.messages(session.id).at(-1);
  assert.equal(message.content, '完成'); assert.equal(message.steps[0].type, 'Observation'); assert.equal(message.toolCalls[0].tool_name, 'search_web'); assert.equal(message.sources[0].source, '文档'); assert.equal(message.context.prompt_version, '001.1'); assert.equal(message.memory, '偏好简洁回答'); assert.equal(message.promptTrace.length, 1);
});

test('chat, tool, rag and react routes all reach a completed message', async () => {
  const transport = new FakeTransport(); const app = new AppController({ repository: new MemoryRepository(), transport });
  for (const mode of ['chat', 'tool', 'rag', 'react']) {
    const session = app.newSession(mode); const running = app.run(session.id, mode);
    transport.event(session.id, { event: 'route', data: { mode } }); transport.event(session.id, { event: 'token', data: { content: mode } }); transport.finish(session.id, { answer: mode, mode, sessionId: session.id, interrupted: false }); await running;
    assert.equal(app.messages(session.id).at(-1).status, 'completed'); assert.equal(app.messages(session.id).at(-1).content, mode);
  }
});

test('session cancellation waits for the old stream and never targets another session', async () => {
  const transport = new FakeTransport(); const app = new AppController({ repository: new MemoryRepository(), transport });
  const a = app.newSession('A'); const b = app.newSession('B'); const runningA = app.run(a.id, '长任务'); const runningB = app.run(b.id, '另一个任务');
  const stopping = app.stop(a.id); await Promise.resolve(); assert.deepEqual(transport.cancelled, [a.id]); assert.equal(app.runs.get(a.id).state, 'stopping'); assert.equal(app.runs.get(b.id).state, 'sending');
  transport.finish(a.id, { answer: '', sessionId: a.id, interrupted: true }); await stopping; await runningA;
  transport.finish(b.id, { answer: 'B 完成', sessionId: b.id, interrupted: false }); await runningB;
  assert.equal(app.messages(a.id).at(-1).status, 'interrupted'); assert.equal(app.messages(b.id).at(-1).status, 'completed');
});

test('failed cancellation restores the active state and still allows the original stream to settle', async () => {
  const transport = new FakeTransport(); transport.cancel = () => Promise.reject(new Error('cancel unavailable'));
  const app = new AppController({ repository: new MemoryRepository(), transport }); const session = app.newSession(); const running = app.run(session.id, '长任务');
  transport.event(session.id, { event: 'token', data: { content: '部分' } }); await assert.rejects(() => app.stop(session.id), /cancel unavailable/); assert.equal(app.runs.get(session.id).state, 'streaming');
  transport.finish(session.id, { answer: '部分完成', sessionId: session.id, interrupted: false }); await running; assert.equal(app.messages(session.id).at(-1).status, 'completed');
});

test('network loss preserves partial text as unknown and prevents automatic replay', async () => {
  const transport = new FakeTransport(); const app = new AppController({ repository: new MemoryRepository(), transport }); const session = app.newSession();
  const running = app.run(session.id, '问题'); transport.event(session.id, { event: 'token', data: { content: '部分回答' } }); const error = Object.assign(new Error('offline'), { code: 'network_error' }); transport.fail(session.id, error);
  await assert.rejects(running);
  assert.equal(app.messages(session.id).at(-1).content, '部分回答'); assert.equal(app.messages(session.id).at(-1).status, 'unknown');
  await assert.rejects(() => app.run(session.id, '不要自动重试'), next => next.code === 'unknown_run');
});

test('document and upload state uses IDs, retains stale data on failure, and separates same names', async () => {
  const transport = new FakeTransport(); const app = new AppController({ repository: new MemoryRepository(), transport });
  await app.loadDocuments(); assert.deepEqual(app.documents.items.map(row => row.id), ['d1', 'd2']); assert.equal(app.documents.items[0].title, app.documents.items[1].title);
  transport.failDocuments = true; await app.loadDocuments(); assert.deepEqual(app.documents.items.map(row => row.id), ['d1', 'd2']); assert.match(app.documents.error, /暂不可用/);
  transport.failDocuments = false; await app.openDocument('d1'); await app.ingestDocument(app.documents.selected.document, app.documents.selected.version); assert.equal(app.documents.items[0].indexStatus, 'indexed');
  const files = [{ name: '同名.md', marker: 'a' }, { name: '同名.md', marker: 'b' }, { name: 'scan.pdf', marker: 'scan' }]; await app.uploadFiles(files);
  assert.equal(app.uploads.length, 3); assert.deepEqual(new Set(app.uploads.slice(0, 2).map(job => job.documentId)), new Set(['uploaded-a', 'uploaded-b'])); assert.equal(app.uploads[2].status, 'needs_ocr');
});

test('document-producing tool completion invalidates the document list once the run settles', async () => {
  const transport = new FakeTransport(); const app = new AppController({ repository: new MemoryRepository(), transport }); const session = app.newSession();
  const running = app.run(session.id, '生成报告', { selectedTools: ['write_document'] }); transport.event(session.id, { event: 'tool_call', data: { tool_name: 'write_document', tool_result: '{}', success: true } }); transport.finish(session.id, { answer: '已生成', sessionId: session.id, interrupted: false }); await running;
  await new Promise(resolve => setTimeout(resolve, 0)); assert.equal(transport.documentLoads, 1);
});

test('artifact links require an explicit document object and are never guessed from answer text', () => {
  assert.deepEqual(normalizeResponse({ answer: '已保存 doc_fake' }).artifacts, []);
  assert.deepEqual(normalizeResponse({ answer: '已保存', document: { id: 'doc_1', title: '报告' } }).artifacts, [{ documentId: 'doc_1', title: '报告' }]);
});

test('tools remove stale selections and registration validates the placeholder endpoint', async () => {
  const transport = new FakeTransport(); const app = new AppController({ repository: new MemoryRepository(), transport }); const session = app.newSession(); session.selectedTools = ['search_web', 'removed_tool'];
  await app.loadTools(); assert.deepEqual(session.selectedTools, ['search_web']); assert.match(app.repository.notice, /removed_tool/);
  await assert.rejects(() => app.registerTool({ name: '3bad', endpoint: 'https://example.com' }), /工具名称/);
  await assert.rejects(() => app.registerTool({ name: 'ok', endpoint: 'file:///tmp/x' }), /http/);
  await app.registerTool({ name: 'custom_api', description: '占位工具', endpoint: 'https://example.com/tool' }); assert.equal(transport.registered.name, 'custom_api'); assert.equal(app.tools.items.at(-1).isMcp, true);
});

class MemoryStorage {
  constructor() { this.values = new Map(); this.fail = false; }
  getItem(key) { return this.values.get(key) ?? null; }
  setItem(key, value) { if (this.fail) throw new Error('QuotaExceededError'); this.values.set(key, value); }
}

test('versioned storage restores active messages as unknown and protects unknown schema', () => {
  const storage = new MemoryStorage(); const workspace = emptyWorkspace(); workspace.sessions.push({ id: 's1', title: 'A', createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), messageIds: ['m1'], draft: '', useRag: false, selectedTools: [] }); workspace.messages.push({ id: 'm1', sessionId: 's1', role: 'assistant', content: '部分', format: 'text', status: 'streaming', steps: [], sources: [], artifacts: [], createdAt: new Date().toISOString() });
  storage.setItem(STORAGE_KEY, JSON.stringify(workspace)); const repo = new WorkspaceRepository(storage); assert.equal(repo.load().messages[0].status, 'unknown');
  const raw = '{"schemaVersion":99,"valuable":"keep"}'; storage.setItem(STORAGE_KEY, raw); const protectedRepo = new WorkspaceRepository(storage); assert.equal(protectedRepo.load().sessions.length, 0); assert.equal(protectedRepo.mode, 'protected'); assert.equal(protectedRepo.save(emptyWorkspace()), false); assert.equal(storage.getItem(STORAGE_KEY), raw);
});

test('legacy migration is idempotent, retains original keys, and refuses partial corrupt data', () => {
  const storage = new MemoryStorage(); const legacy = JSON.stringify([{ id: 'legacy-1', title: '旧会话', ts: 10, messages: [] }]); storage.setItem('ai_sessions', legacy); storage.setItem('ai_docs', '[]');
  const first = new WorkspaceRepository(storage).load(); assert.equal(first.sessions[0].id, 'legacy-1'); assert.equal(storage.getItem('ai_sessions'), legacy); assert.ok(storage.getItem(STORAGE_KEY));
  storage.setItem('ai_sessions', '{changed but broken'); const second = new WorkspaceRepository(storage).load(); assert.equal(second.sessions[0].id, 'legacy-1');
  const broken = new MemoryStorage(); broken.setItem('ai_sessions', legacy); broken.setItem('ai_docs', '{broken'); const repo = new WorkspaceRepository(broken); assert.equal(repo.load().sessions.length, 0); assert.equal(repo.mode, 'protected'); assert.equal(broken.getItem(STORAGE_KEY), null); assert.equal(broken.getItem('ai_sessions'), legacy);
});

test('quota failure keeps the last good value and raw export includes every legacy source', () => {
  const storage = new MemoryStorage(); const repo = new WorkspaceRepository(storage); const workspace = repo.load(); const good = storage.getItem(STORAGE_KEY); workspace.sessions.push({ id: 'new', title: 'A', createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), messageIds: [], draft: '', useRag: false, selectedTools: [] });
  storage.fail = true; assert.equal(repo.save(workspace), false); assert.equal(storage.getItem(STORAGE_KEY), good); storage.fail = false;
  const raw = JSON.parse(repo.exportRaw('http://local')); assert.ok(Object.hasOwn(raw.data, 'ai_sessions')); assert.ok(Object.hasOwn(raw.data, 'ai_docs')); assert.equal(repo.mode, 'memory');
});
