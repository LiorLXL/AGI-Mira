/** Network transport and backend-field adapters. This module never touches the DOM or storage. */
export class ApiError extends Error {
  constructor(message, status = 0, body = null, code = 'http_error') { super(message); this.name = 'ApiError'; this.status = status; this.body = body; this.code = code; }
}
export class ProtocolError extends ApiError { constructor(message, body = null) { super(message, 200, body, 'protocol_error'); this.name = 'ProtocolError'; } }
export class StreamEOFError extends ApiError { constructor() { super('流在有效 done 事件前结束', 200, null, 'eof'); this.name = 'StreamEOFError'; } }

const KNOWN_EVENTS = new Set(['start', 'route', 'token', 'step', 'tool_call', 'rag_result', 'memory', 'context', 'done']);
const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const array = value => Array.isArray(value) ? value : [];

async function readBody(response) {
  const raw = await response.text();
  if (!raw) return { data: null, raw: '' };
  try { return { data: JSON.parse(raw), raw }; } catch (_) { return { data: null, raw }; }
}

export async function requestJSON(path, options = {}) {
  const headers = { Accept: 'application/json', ...(options.headers || {}) };
  if (options.body && !(options.body instanceof FormData) && !headers['Content-Type']) headers['Content-Type'] = 'application/json';
  let response;
  try { response = await fetch(path, { ...options, headers }); }
  catch (error) { throw new ApiError(error?.message || '网络连接失败', 0, null, error?.name === 'AbortError' ? 'aborted' : 'network_error'); }
  const { data, raw } = await readBody(response);
  if (!response.ok) {
    const detail = data?.detail;
    const message = typeof detail === 'string' ? detail : Array.isArray(detail) ? detail.map(item => item?.msg).filter(Boolean).join('；') : data?.error || data?.message || raw || response.statusText || `HTTP ${response.status}`;
    throw new ApiError(String(message), response.status, data ?? raw);
  }
  if (data === null && raw) throw new ProtocolError('服务返回了无法解析的数据', raw);
  return data;
}

export function normalizeSearchResults(value) {
  const rows = Array.isArray(value) ? value : array(value?.search_results ?? value?.results);
  return rows.map((row, index) => {
    const scoreValue = row?.score ?? row?.similarity;
    const score = Number.isFinite(Number(scoreValue)) ? Number(scoreValue) : null;
    return {
      ...row, id: String(row?.id || `source-${index}`),
      content: String(row?.content ?? row?.chunk?.content ?? row?.text ?? ''),
      source: String(row?.source ?? row?.filename ?? row?.title ?? '来源'), score,
    };
  });
}

function artifactFrom(value) {
  if (!isObject(value) || !value.id) return null;
  return { documentId: String(value.id), title: String(value.title || '生成文档') };
}

export function normalizeResponse(body) {
  const value = isObject(body) ? body : {};
  const artifacts = [artifactFrom(value.document), artifactFrom(value.task?.document), artifactFrom(value.tool_call?.document)].filter(Boolean);
  return {
    ...value, answer: typeof value.answer === 'string' ? value.answer : '', mode: typeof value.mode === 'string' ? value.mode : '',
    sessionId: value.session_id ?? value.sessionId ?? null, requestId: value.request_id ?? value.requestId ?? null,
    steps: array(value.steps), toolCalls: value.tool_call ? [value.tool_call] : array(value.toolCalls),
    searchResults: normalizeSearchResults(value.search_results ?? value.searchResults), artifacts,
    contextTrace: array(value.context_trace), promptTrace: array(value.prompt_trace), interrupted: value.interrupted === true,
    success: value.success,
  };
}

export function normalizeEvent(name, payload) {
  if (!KNOWN_EVENTS.has(name)) return { event: 'unknown', name, payload };
  if (!isObject(payload)) throw new ProtocolError(`${name} 事件不是对象`, payload);
  if (name === 'token') return { event: name, data: { content: String(payload.content ?? '') } };
  if (name === 'rag_result') return { event: name, data: { ...payload, searchResults: normalizeSearchResults(payload.search_results ?? payload.results ?? payload) } };
  if (name === 'done') return { event: name, data: { response: normalizeResponse(payload) } };
  return { event: name, data: { ...payload } };
}

function parseBlock(block) {
  let name = 'message';
  const dataLines = [];
  for (const line of block.split('\n')) {
    if (!line || line.startsWith(':')) continue;
    if (line.startsWith('event:')) name = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).replace(/^ /, ''));
  }
  if (!dataLines.length) return null;
  const raw = dataLines.join('\n');
  if (raw === '[DONE]') return { event: 'transport_end', terminal: true };
  if (!KNOWN_EVENTS.has(name)) {
    let payload = raw;
    try { payload = JSON.parse(raw); } catch (_) {}
    return { event: 'unknown', name, payload };
  }
  let payload;
  try { payload = JSON.parse(raw); } catch (_) { throw new ProtocolError(`${name} 事件包含无效 JSON`, raw); }
  return normalizeEvent(name, payload);
}

async function* chunks(stream) {
  if (!stream?.getReader) throw new ProtocolError('浏览器不支持读取流式响应');
  const reader = stream.getReader();
  try { while (true) { const item = await reader.read(); if (item.done) break; yield item.value; } }
  finally { reader.releaseLock?.(); }
}

export async function* parseSSE(stream) {
  const decoder = new TextDecoder();
  let buffer = '';
  for await (const chunk of chunks(stream)) {
    buffer += decoder.decode(chunk, { stream: true });
    buffer = buffer.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    const blocks = buffer.split('\n\n');
    buffer = blocks.pop() || '';
    for (const block of blocks) { const event = parseBlock(block); if (event) yield event; }
  }
  buffer += decoder.decode();
  if (buffer.trim()) { const event = parseBlock(buffer); if (event) yield event; }
}

export async function streamChat(payload, onEvent = () => {}, signal) {
  let response;
  try { response = await fetch('/api/chat/stream', { method: 'POST', headers: { Accept: 'text/event-stream', 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal }); }
  catch (error) { if (error?.name === 'AbortError') throw error; throw new ApiError(error?.message || '流式请求失败', 0, null, 'network_error'); }
  if (!response.ok) { const { data, raw } = await readBody(response); throw new ApiError(String(data?.detail || data?.error || raw || '流式请求失败'), response.status, data ?? raw); }
  let done = null;
  for await (const event of parseSSE(response.body)) {
    if (event.event === 'done') done = event.data.response;
    onEvent(event);
  }
  if (!done) throw new StreamEOFError();
  return done;
}

export const chat = payload => requestJSON('/api/chat', { method: 'POST', body: JSON.stringify(payload) }).then(normalizeResponse);
export const cancel = sessionId => {
  if (typeof sessionId !== 'string' || !sessionId.trim()) return Promise.reject(new ApiError('session_id 不能为空', 422, null, 'validation_error'));
  return requestJSON('/api/chat/cancel', { method: 'POST', body: JSON.stringify({ session_id: sessionId }) });
};
export const listDocuments = () => requestJSON('/api/documents').then(body => array(body?.documents).map(normalizeDocument));
export const getDocument = id => requestJSON(`/api/documents/${encodeURIComponent(id)}`).then(normalizeDocumentDetail);
export const ingestDocument = (id, versionId = '') => requestJSON(`/api/documents/${encodeURIComponent(id)}/ingest`, { method: 'POST', body: JSON.stringify(versionId ? { version_id: versionId } : {}) });
export const uploadFile = file => { const form = new FormData(); form.append('file', file); return requestJSON('/api/upload', { method: 'POST', body: form }); };
export const listTools = () => requestJSON('/api/tools').then(body => array(body).map(normalizeTool));
export const registerTool = value => requestJSON('/api/tools/mcp', { method: 'POST', body: JSON.stringify(value) });
export const getHealth = () => requestJSON('/health');

export function normalizeDocument(value) {
  const row = isObject(value) ? value : {};
  return {
    ...row, id: String(row.id || ''), title: String(row.title || '未命名文档'), source: String(row.source || ''),
    docType: String(row.doc_type || ''), status: String(row.status || ''), createdBy: String(row.created_by || ''),
    latestVersion: Number.isFinite(Number(row.latest_version)) ? Number(row.latest_version) : 0,
    latestVersionId: String(row.latest_version_id || ''), metadata: isObject(row.latest_metadata) ? row.latest_metadata : {},
    contentChars: Number(row.latest_content_chars || 0), parser: String(row.latest_parser || ''), indexStatus: 'unknown',
  };
}

export function normalizeDocumentDetail(body) {
  const document = normalizeDocument(body?.document);
  const version = isObject(body?.version) ? body.version : {};
  return {
    document, version: {
      ...version, id: String(version.id || ''), documentId: String(version.document_id || document.id),
      version: Number(version.version || document.latestVersion || 0), content: String(version.content_md || ''),
      summary: String(version.summary || ''), metadata: isObject(version.metadata) ? version.metadata : {},
    },
  };
}

export function normalizeTool(value) {
  const row = isObject(value) ? value : {};
  return { name: String(row.name || ''), description: String(row.description || ''), params: array(row.params), isMcp: row.is_mcp === true };
}
