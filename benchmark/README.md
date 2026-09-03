# AGI-Mira × Milo-bench

AGI-Mira 是唯一被测系统。Milo-bench 提供固定语料与 qrels，
`evaluate_milo.py` 只通过 HTTP 导入、检索和评分。

LLM 与 Embedding 配置统一来自 `config/config.local.yaml`。benchmark API
不会接收 API Key。客户端首次运行时会自动生成 Git 忽略的
`benchmark/.benchmark-token`，后续自动复用。

完整的首次启动、环境切换、评测命令和故障处理请查看：

- [`docs/AGI_MIRA_RUNBOOK.md`](../docs/AGI_MIRA_RUNBOOK.md)

最短流程：

```powershell
docker compose down
docker compose -p agi-mira-bench up -d --build

.\.venv\Scripts\python.exe benchmark\evaluate_milo.py configure
.\.venv\Scripts\python.exe benchmark\evaluate_milo.py prepare `
  --confirm-embedding-cost `
  --embedding-workers 5
.\.venv\Scripts\python.exe benchmark\evaluate_milo.py evaluate --query-limit 3
```

对于有效 benchmark，当前 project 的 `rag_chunks` 必须只包含 Milo 数据。
普通开发和 benchmark 使用不同 Compose project name 与数据卷，不要同时启动。
