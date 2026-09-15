# Mira 原生前端

无需安装 Node、npm 或前端依赖。项目启动后，FastAPI 同源提供 HTML、CSS、JavaScript 和 `/api/*`。

```powershell
python main.py
```

打开 `http://localhost:8090/`。页面使用 hash 导航：`#/chat/{sessionId}`、`#/documents/{documentId}`、`#/settings`。

## 文件职责

- `index.html`：语义页面和弹窗结构。
- `styles.css`：浅色/深色主题、工作区样式和响应式布局。
- `app.js`：会话、请求生命周期、文档/工具状态和页面协调。
- `api.js`：HTTP、POST SSE、错误与后端字段转换；不操作 DOM。
- `storage.js`：版本化本机存储、旧数据迁移和备份；不发送请求。
- `ui.js`：安全 DOM/Markdown 渲染；不调用 fetch 或 localStorage。
- `index.legacy.html`：重构前页面快照，可直接访问 `/index.legacy.html`。

## 本机历史与迁移

新版使用 `mira_workspace_v2`，原来的 `ai_sessions` 和 `ai_docs` 不删除、不覆盖。首次同源打开时会迁移旧会话 ID、顺序和可读消息；旧 HTML 在未挂入页面的 `<template>` 中解析，移除活动元素及全部属性，再保存纯文本。

刷新时，未完成的 `sending/streaming/stopping` 消息会变为 `unknown`，保留已生成正文且不会自动重发。unknown 会话需要新建会话继续。损坏 JSON、未知版本或引用不一致进入保护模式，不覆盖原值；存储受限时使用内存模式并给出提示。

设置页提供当前工作区和完整原始存储的导出。导入只允许在没有会话、消息和上传记录的空白工作区执行；导入文件另存为 `mira_workspace_import_backup`。本机导入不会重建后端会话上下文。

浏览器存储按 origin 隔离；协议、主机或端口变化时，需要先在旧 origin 导出，再到空白新 origin 导入。

## 文档与工具

文档中心以服务端 `document.id` 为主键，同名文档不会合并。上传任务单独显示；`needs_ocr`、解析成功、版本和索引状态分别呈现。重新入库只有在返回明确的 `indexed_count/chunk_count` 后显示已更新。

文档列表请求失败时保留上一份有效结果。只有响应明确包含 document ID 才生成文档链接。界面不提供后端尚未保证的硬删除、版本历史或 OCR 操作。

工具选择按会话保存；后端列表移除工具时会清理失效选择并提示。设置页保留工具登记，但明确标注当前是占位实现，登记成功不表示远程 endpoint 已连通。

## 验证

```powershell
node --check frontend/api.js
node --check frontend/storage.js
node --check frontend/ui.js
node --check frontend/app.js
node --test tests/frontend_runtime.test.mjs

.\.venv\Scripts\python.exe -m pytest `
  tests/test_frontend_p0_contracts.py `
  tests/test_frontend_main_alignment.py `
  tests/test_session_isolation.py `
  tests/test_document_api.py -q
```

`tests/frontend_fixture_server.py` 仅供浏览器验收，使用确定性 Agent 替身，不访问模型或数据库。应用正常运行不依赖 Node 或该测试服务。

## 静态交付验证

`tests/test_frontend_delivery.py` 检查正式 HTML/CSS/ES Modules、legacy、Swagger、OpenAPI、benchmark 路由、404 行为和 MIME，同时锁定 Dockerfile/Compose 的纯 Python 静态交付方式。

真实入口可用禁用外部基础设施的配置独立验证：

```powershell
$env:AGI_CONFIG = "$PWD/tests/fixtures/frontend-delivery-config.yaml"
.\.venv\Scripts\python.exe main.py
```

该配置监听 8092，只用于检查启动和静态/API 路由，不访问模型或数据库。旧版回退页为 `http://localhost:8092/index.legacy.html`，和新版共用 origin，因此 `ai_sessions`、`ai_docs` 与 `mira_workspace_v2` 可同时保留。

Dockerfile 直接 `COPY frontend/ ./frontend/`，运行镜像继续使用 Python 入口，没有 Node 构建阶段或新增前端服务。当前验证环境没有 Docker/Podman/nerdctl/buildah，因此镜像实际构建与 `docker compose up` 仍需在具备容器运行时的机器执行。
