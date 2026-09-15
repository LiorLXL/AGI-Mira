"""Feature 002 T502/T503: static delivery, route ordering and Docker contract."""
import asyncio
from pathlib import Path

from config.config import APIConfig
from internal.handler.benchmark import setup_benchmark_routes
from internal.handler.handler import setup_routes
from test_frontend_p0_contracts import ContractAgent, _Infra


ROOT = Path(__file__).parents[1]
FRONTEND = ROOT / "frontend"


def request(app, path):
    async def run():
        sent = False
        messages = []

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            messages.append(message)

        await app(
            {
                "type": "http", "asgi": {"version": "3.0"},
                "http_version": "1.1", "method": "GET", "path": path,
                "raw_path": path.encode(), "query_string": b"", "headers": [],
                "client": ("delivery", 1), "server": ("delivery", 80), "scheme": "http",
            }, receive, send,
        )
        start = next(item for item in messages if item["type"] == "http.response.start")
        headers = {key.decode().lower(): value.decode() for key, value in start.get("headers", [])}
        body = b"".join(item.get("body", b"") for item in messages if item["type"] == "http.response.body")
        return start["status"], headers, body

    return asyncio.run(run())


def delivery_app(monkeypatch):
    monkeypatch.setenv("FRONTEND_DIR", str(FRONTEND))
    agent = ContractAgent()
    infra = _Infra(agent)
    agent.inf = infra
    cfg = APIConfig()
    app = setup_routes(agent, infra, cfg)
    setup_benchmark_routes(app, cfg)
    return app


def test_static_assets_have_correct_mime_and_missing_asset_is_real_404(monkeypatch):
    app = delivery_app(monkeypatch)
    expected = {
        "/": ("text/html", "src=\"app.js?v=002.1\""),
        "/styles.css": ("text/css", "Application workspace"),
        "/app.js": (("text/javascript", "application/javascript"), "class AppController"),
        "/api.js": (("text/javascript", "application/javascript"), "streamChat"),
        "/storage.js": (("text/javascript", "application/javascript"), "WorkspaceRepository"),
        "/ui.js": (("text/javascript", "application/javascript"), "renderDocumentReader"),
        "/index.legacy.html": ("text/html", "function send()"),
    }
    for path, (content_type, marker) in expected.items():
        status, headers, body = request(app, path)
        assert status == 200, path
        accepted = content_type if isinstance(content_type, tuple) else (content_type,)
        assert headers["content-type"].startswith(accepted), (path, headers)
        assert marker.encode() in body, path

    status, headers, _body = request(app, "/missing-static-file.js")
    assert status == 404
    assert headers["content-type"].startswith("application/json")


def test_api_swagger_and_benchmark_routes_are_not_shadowed_by_static_mount(monkeypatch):
    app = delivery_app(monkeypatch)
    checks = {
        "/health": (200, "application/json"),
        "/api/status": (200, "application/json"),
        "/docs": (200, "text/html"),
        "/openapi.json": (200, "application/json"),
        "/api/benchmark/status": (200, "application/json"),
    }
    for path, (expected_status, content_type) in checks.items():
        status, headers, _body = request(app, path)
        assert status == expected_status, path
        assert headers["content-type"].startswith(content_type), (path, headers)


def test_docker_delivery_copies_plain_frontend_without_node_build_or_extra_service():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    required = ["index.html", "styles.css", "app.js", "api.js", "storage.js", "ui.js", "index.legacy.html"]

    assert "COPY frontend/ ./frontend/" in dockerfile
    assert "FRONTEND_DIR=/app/frontend" in dockerfile
    assert "ENTRYPOINT [\"python\", \"-u\", \"main.py\"]" in dockerfile
    assert "FROM node" not in dockerfile and "npm " not in dockerfile and "yarn " not in dockerfile
    assert "FRONTEND_DIR: /app/frontend" in compose
    assert "frontend:" not in compose and "node:" not in compose
    assert "frontend/" not in dockerignore and "*.js" not in dockerignore and "*.css" not in dockerignore and "*.html" not in dockerignore
    assert all((FRONTEND / name).is_file() for name in required)


def test_main_keeps_frontend_path_independent_from_current_working_directory():
    main = (ROOT / "main.py").read_text(encoding="utf-8")
    assert 'os.path.join(PROJECT_ROOT, "frontend")' in main
    assert 'os.environ.setdefault("FRONTEND_DIR"' in main
