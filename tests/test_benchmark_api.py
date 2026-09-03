import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount

from internal.handler.benchmark import (
    BenchmarkController,
    BenchmarkRuntime,
    ConfigureRequest,
    benchmark_key,
    chunk_id_from_key,
    configure_runtime_config,
    setup_benchmark_routes,
)
from internal.rag.rewriter import LLMRewriter


class BenchmarkAPITest(unittest.TestCase):
    def test_chunk_id_round_trip(self):
        chunk_id = "juejin_123_00001"

        self.assertEqual(chunk_id_from_key(benchmark_key(chunk_id)), chunk_id)
        self.assertIsNone(chunk_id_from_key("ordinary-document"))

    def test_container_configuration_uses_compose_service_names(self):
        base = SimpleNamespace(
            embedding_api_url="https://example.test/embeddings",
            embedding_api_key="embedding-secret",
            embedding_model="embedding-model",
            rag_milvus_dim=1024,
            llm_api_url="",
            llm_api_key="",
            llm_model="",
            temperature=0.7,
            pg_host="localhost",
            pg_port=5432,
            milvus_host="localhost",
            milvus_port=19530,
            es_addresses=["http://localhost:9200"],
            es_username="elastic",
            es_password="password",
            kafka_brokers=["localhost:29092"],
            chunk_size=200,
            chunk_overlap=50,
            top_k=3,
            rrf_constant_k=1,
            semantic_weight=0.1,
            enable_hybrid_search=False,
            rag_rewrite_enabled=True,
            rag_rerank_enabled=True,
            kg_enabled=True,
            neo4j_uri="bolt://localhost:7687",
            sandbox_enabled=True,
        )
        request = ConfigureRequest.model_validate(
            {
                "embedding_dim": 2048,
                "benchmark_token": "benchmark-token-123456",
            }
        )

        cfg = configure_runtime_config(base, request, running_in_container=True)

        self.assertEqual(cfg.pg_host, "postgres")
        self.assertEqual(cfg.milvus_host, "milvus")
        self.assertEqual(cfg.es_addresses, ["http://elasticsearch:9200"])
        self.assertEqual(cfg.rag_milvus_dim, 2048)
        self.assertEqual(cfg.chunk_size, 400)
        self.assertEqual(cfg.chunk_overlap, 80)
        self.assertEqual(cfg.top_k, 10)
        self.assertFalse(cfg.kg_enabled)
        self.assertFalse(cfg.rag_rewrite_enabled)
        self.assertFalse(cfg.rag_rerank_enabled)
        self.assertEqual(cfg.embedding_api_key, "embedding-secret")
        self.assertEqual(cfg.embedding_api_url, "https://example.test/embeddings")
        self.assertEqual(cfg.embedding_model, "embedding-model")

    def test_status_is_safe_before_configuration(self):
        controller = BenchmarkController(SimpleNamespace())

        self.assertEqual(controller.status(), {"configured": False})

    def test_finalize_supports_milvus_client_without_flush(self):
        class Indices:
            def refresh(self, index):
                self.index = index

        class ES:
            indices = Indices()

        class Milvus:
            def __init__(self):
                self.loaded = False

            def load_collection(self, collection_name):
                self.loaded = collection_name == "rag_chunks"

            def query(self, **_kwargs):
                return [{"count(*)": 23493}]

        inf = SimpleNamespace(_es=ES(), _milvus=Milvus())
        runtime = BenchmarkRuntime(SimpleNamespace(), inf, SimpleNamespace())
        controller = BenchmarkController(SimpleNamespace())
        controller._runtime = runtime
        controller._require_isolated_corpus = lambda _runtime: None
        controller._pg_counts = lambda _runtime: (23493, 23493, 23493)
        controller.status = lambda: {"counts": {"milvus": 23493}}

        result = controller.finalize()

        self.assertTrue(inf._milvus.loaded)
        self.assertEqual(result["counts"]["milvus"], 23493)

    def test_enhanced_search_requires_chat_completions_url(self):
        cfg = SimpleNamespace(
            llm_api_url="https://provider.example/v1/usage",
            is_real_llm=lambda: True,
        )
        runtime = BenchmarkRuntime(cfg, SimpleNamespace(), SimpleNamespace())
        controller = BenchmarkController(SimpleNamespace())
        controller._runtime = runtime
        controller._require_isolated_corpus = lambda _runtime: None

        with self.assertRaisesRegex(HTTPException, "Chat Completions"):
            controller.search(
                SimpleNamespace(query="question", top_k=10, rewrite=True, rerank=True)
            )

    def test_strict_rewriter_exposes_invalid_model_output(self):
        rewriter = LLMRewriter(lambda _system, _user: "not-json", strict=True)

        with self.assertRaisesRegex(RuntimeError, "raw_preview='not-json'"):
            rewriter.rewrite("question", [])

    @patch("internal.handler.benchmark.requests.post")
    def test_benchmark_chat_requests_enough_output_tokens(self, post):
        post.return_value = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"queries":["a","b","c"]}'},
                    }
                ]
            },
        )
        cfg = SimpleNamespace(
            llm_api_url="https://provider.example/v1/chat/completions",
            llm_api_key="secret",
            llm_model="deepseek",
        )
        runtime = BenchmarkRuntime(cfg, SimpleNamespace(), SimpleNamespace())

        content = BenchmarkController._strict_generate(runtime, "system", "user")

        self.assertEqual(content, '{"queries":["a","b","c"]}')
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertEqual(payload["temperature"], 0.0)
        self.assertFalse(payload["stream"])


    def test_benchmark_authorization_rejects_missing_token(self):
        controller = BenchmarkController(SimpleNamespace())
        controller._token_hash = controller._hash_token("benchmark-token-123456")

        with self.assertRaisesRegex(Exception, "invalid benchmark token"):
            controller.authorize(None)

    def test_benchmark_routes_are_inserted_before_frontend_mount(self):
        app = FastAPI()
        app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

        setup_benchmark_routes(app, SimpleNamespace())

        benchmark_index = next(
            index
            for index, route in enumerate(app.router.routes)
            if getattr(route, "path", None) == "/api/benchmark/status"
        )
        mount_index = next(
            index for index, route in enumerate(app.router.routes) if isinstance(route, Mount)
        )
        self.assertLess(benchmark_index, mount_index)


if __name__ == "__main__":
    unittest.main()
