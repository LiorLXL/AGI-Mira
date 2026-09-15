export const STORAGE_KEY = 'mira_workspace_v2';
export const LEGACY_SESSION_KEY = 'ai_sessions';
export const LEGACY_DOCS_KEY = 'ai_docs';
export const IMPORT_BACKUP_KEY = 'mira_workspace_import_backup';
const ACTIVE_STATES = new Set(['sending', 'streaming', 'stopping']);
const MESSAGE_STATES = new Set(['sending', 'streaming', 'stopping', 'completed', 'failed', 'interrupted', 'unknown']);

const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const string = (value, fallback = '') => typeof value === 'string' ? value : fallback;
const isoTime = (value, fallback = new Date().toISOString()) => {
  if (typeof value === 'number' && Number.isFinite(value)) return new Date(value).toISOString();
  if (typeof value === 'string' && value) return value;
  return fallback;
};
const validId = value => typeof value === 'string' && value === value.trim() && value.length >= 1 && value.length <= 128;

/** Parse legacy HTML inside an inert template. Nodes never enter the live document. */
export function legacyText(html, owner = globalThis.document) {
  if (!owner?.createElement) throw new Error('HTML migration requires a browser document');
  const template = owner.createElement('template');
  template.innerHTML = String(html ?? '');
  template.content.querySelectorAll('script,style,iframe,object,embed,template,noscript,svg,math,link,meta').forEach(node => node.remove());
  template.content.querySelectorAll('*').forEach(node => [...node.attributes].forEach(attribute => node.removeAttribute(attribute.name)));
  return String(template.content.textContent || '').replace(/\n{3,}/g, '\n\n').trim();
}

export function emptyWorkspace() {
  return { schemaVersion: 2, sessions: [], messages: [], uploads: [], documents: [], updatedAt: new Date().toISOString() };
}

function normalizeMessage(value, sessionId, index) {
  if (!isObject(value)) throw new Error('Invalid message record');
  const roleValue = value.role === 'ai' ? 'assistant' : value.role;
  const statusValue = string(value.status, 'completed');
  return {
    id: string(value.id, `legacy:${sessionId}:${index}`), sessionId,
    role: ['user', 'assistant', 'system'].includes(roleValue) ? roleValue : 'assistant',
    content: typeof value.content === 'string' ? value.content : typeof value.html === 'string' ? legacyText(value.html) : '',
    format: value.format === 'markdown' ? 'markdown' : 'text',
    status: MESSAGE_STATES.has(statusValue) ? (ACTIVE_STATES.has(statusValue) ? 'unknown' : statusValue) : 'completed',
    steps: Array.isArray(value.steps) ? value.steps : [], toolCalls: Array.isArray(value.toolCalls) ? value.toolCalls : [],
    sources: Array.isArray(value.sources) ? value.sources : [], artifacts: Array.isArray(value.artifacts) ? value.artifacts : [],
    context: isObject(value.context) ? value.context : null, promptTrace: Array.isArray(value.promptTrace) ? value.promptTrace : [], memory: string(value.memory),
    createdAt: isoTime(value.createdAt || value.created_at || value.ts), completedAt: value.completedAt ? isoTime(value.completedAt) : undefined,
    migrated: Boolean(value.migrated || typeof value.html === 'string'),
  };
}

function normalizeSession(value, index, fallbackTime) {
  if (!isObject(value) || !validId(value.id)) throw new Error(`Invalid session at index ${index}`);
  const rawTitle = string(value.title, '新对话');
  return {
    id: value.id, title: (rawTitle.includes('<') ? legacyText(rawTitle) : rawTitle) || '新对话',
    createdAt: isoTime(value.createdAt || value.created_at || value.ts, fallbackTime), updatedAt: isoTime(value.updatedAt || value.updated_at || value.ts, fallbackTime),
    messageIds: Array.isArray(value.messageIds) ? value.messageIds.map(String) : [], draft: string(value.draft),
    useRag: Boolean(value.useRag ?? value.use_rag), selectedTools: Array.isArray(value.selectedTools) ? value.selectedTools.map(String) : [],
    migrated: Boolean(value.migrated || Array.isArray(value.messages)),
  };
}

function normalizeUpload(value, index) {
  if (!isObject(value)) throw new Error(`Invalid upload at index ${index}`);
  const filename = string(value.filename || value.name);
  if (!filename) throw new Error(`Upload ${index} has no filename`);
  return {
    id: string(value.id, `legacy-upload:${index}`), filename,
    status: ['pending', 'uploading', 'completed', 'needs_ocr', 'failed', 'legacy'].includes(value.status) ? value.status : 'legacy',
    error: string(value.error), documentId: string(value.documentId || value.document_id) || undefined,
    versionId: string(value.versionId || value.version_id) || undefined, docHash: string(value.docHash || value.doc_hash) || undefined,
    createdAt: isoTime(value.createdAt || value.ts),
  };
}

export function normalizeWorkspace(value) {
  if (!isObject(value) || !Array.isArray(value.sessions)) throw new Error('Workspace has no session list');
  const version = value.schemaVersion ?? value.version;
  if (version !== 2) throw new Error(`Unsupported workspace version: ${String(version)}`);
  const fallbackTime = new Date().toISOString();
  const sessions = value.sessions.map((session, index) => normalizeSession(session, index, fallbackTime));
  const sessionIds = new Set(sessions.map(session => session.id));
  if (sessionIds.size !== sessions.length) throw new Error('Duplicate session ID');
  const messages = [];
  value.sessions.forEach((rawSession, sessionIndex) => {
    if (Array.isArray(rawSession.messages)) rawSession.messages.forEach((message, index) => messages.push(normalizeMessage(message, sessions[sessionIndex].id, index)));
  });
  if (Array.isArray(value.messages)) value.messages.forEach((message, index) => {
    if (!isObject(message) || !validId(message.sessionId) || !sessionIds.has(message.sessionId)) throw new Error(`Message ${index} has invalid session ownership`);
    messages.push(normalizeMessage(message, message.sessionId, index));
  });
  const byMessageId = new Map();
  messages.forEach(message => { if (byMessageId.has(message.id)) throw new Error('Duplicate message ID'); byMessageId.set(message.id, message); });
  sessions.forEach(session => {
    const explicit = session.messageIds.filter(id => byMessageId.get(id)?.sessionId === session.id);
    const discovered = messages.filter(message => message.sessionId === session.id).map(message => message.id);
    session.messageIds = [...new Set([...explicit, ...discovered])];
  });
  return {
    schemaVersion: 2, sessions, messages,
    uploads: (Array.isArray(value.uploads) ? value.uploads : []).map(normalizeUpload),
    documents: Array.isArray(value.documents) ? value.documents : [], updatedAt: isoTime(value.updatedAt, fallbackTime),
    migration: isObject(value.migration) ? value.migration : undefined,
  };
}

export function migrateLegacySessions(sessionsRaw, docsRaw = null) {
  const sessions = sessionsRaw == null ? [] : JSON.parse(sessionsRaw);
  const docs = docsRaw == null ? [] : JSON.parse(docsRaw);
  if (!Array.isArray(sessions) || !Array.isArray(docs)) throw new Error('Legacy storage must contain arrays');
  const workspace = normalizeWorkspace({ schemaVersion: 2, sessions, messages: [], documents: [] });
  workspace.uploads = docs.map(normalizeUpload);
  workspace.migration = { source: 'legacy', completedAt: new Date().toISOString() };
  return workspace;
}

export class WorkspaceRepository {
  constructor(storage = globalThis.localStorage) {
    this.storage = storage; this.mode = 'memory'; this.notice = storage ? '' : '浏览器存储不可用，当前内容仅保留在内存。'; this.expectedRaw = null;
  }

  load() {
    if (!this.storage) return emptyWorkspace();
    let current, legacy, docs;
    try { current = this.storage.getItem(STORAGE_KEY); legacy = this.storage.getItem(LEGACY_SESSION_KEY); docs = this.storage.getItem(LEGACY_DOCS_KEY); this.expectedRaw = current; }
    catch (_) { this.notice = '无法读取本机存储，当前内容仅保留在内存。'; return emptyWorkspace(); }
    try {
      if (current !== null) { const workspace = normalizeWorkspace(JSON.parse(current)); this.mode = 'saved'; return workspace; }
      const workspace = legacy !== null || docs !== null ? migrateLegacySessions(legacy, docs) : emptyWorkspace();
      this.mode = 'saved'; const persisted = this.save(workspace);
      if (!persisted) this.notice = '旧记录已读取，但尚未写入新版存储。';
      else if (workspace.migration) this.notice = '旧记录已安全迁移；原记录仍保留，后端上下文可能不完整。';
      return workspace;
    } catch (_) { this.mode = 'protected'; this.notice = '检测到损坏数据或未知版本。原始记录已保护，请先导出备份。'; return emptyWorkspace(); }
  }

  save(workspace) {
    if (!this.storage || this.mode === 'protected') return false;
    try {
      if (this.storage.getItem(STORAGE_KEY) !== this.expectedRaw) { this.mode = 'protected'; this.notice = '本机数据已被其他页面更改，已停止覆盖。请导出后刷新。'; return false; }
      const normalized = normalizeWorkspace(workspace); normalized.updatedAt = new Date().toISOString();
      const raw = JSON.stringify(normalized); this.storage.setItem(STORAGE_KEY, raw); this.expectedRaw = raw; this.mode = 'saved'; this.notice = ''; return true;
    } catch (_) { this.mode = 'memory'; this.notice = '本机保存失败，当前更改仅保留在内存；原记录未删除。'; return false; }
  }

  exportRaw(origin = globalThis.location?.origin || '') {
    if (!this.storage) throw new Error('浏览器存储不可读');
    const keys = [STORAGE_KEY, LEGACY_SESSION_KEY, LEGACY_DOCS_KEY, IMPORT_BACKUP_KEY];
    return JSON.stringify({ backupVersion: 1, origin, exportedAt: new Date().toISOString(), data: Object.fromEntries(keys.map(key => [key, this.storage.getItem(key)])) }, null, 2);
  }

  exportWorkspace(workspace, origin = globalThis.location?.origin || '') {
    return JSON.stringify({ backupVersion: 1, origin, exportedAt: new Date().toISOString(), workspace: normalizeWorkspace(workspace) }, null, 2);
  }

  importBackup(serialized, currentWorkspace) {
    if (!this.storage || this.mode === 'protected') throw new Error('当前存储不可写或处于保护状态');
    if (currentWorkspace.sessions.length || currentWorkspace.messages.length || currentWorkspace.uploads.length) throw new Error('请在空白本机工作区导入');
    const backup = JSON.parse(serialized);
    if (!isObject(backup) || backup.backupVersion !== 1) throw new Error('不支持的备份格式');
    let workspace;
    if (backup.workspace) workspace = normalizeWorkspace(backup.workspace);
    else if (isObject(backup.data)) { const current = backup.data[STORAGE_KEY]; workspace = current ? normalizeWorkspace(JSON.parse(current)) : migrateLegacySessions(backup.data[LEGACY_SESSION_KEY], backup.data[LEGACY_DOCS_KEY]); }
    else throw new Error('备份不包含工作区数据');
    this.storage.setItem(IMPORT_BACKUP_KEY, serialized);
    if (!this.save(workspace)) throw new Error(this.notice || '导入保存失败');
    return workspace;
  }
}
