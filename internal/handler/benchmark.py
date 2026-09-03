"""Milo-bench HTTP adapter for AGI-Mira.

The normal chat and document APIs intentionally do not expose stable child
chunk identifiers.  Retrieval benchmarks need those identifiers, immutable
chunk boundaries, and retrieval-only responses.  This module adds that narrow
surface without changing the production RAG flow.

Provider credentials come from AGI-Mira's already-loaded local configuration.
The benchmark API never accepts or returns API keys.  The adapter opens its own
connections so benchmark lifecycle and production request state stay separate.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, SecretStr
import requests
from starlette.routing import Mount

from config.config import APIConfig
from internal.infra.infra import Infrastructure, RAG_COLLECTION
from internal.llm.llm import Client as LLMClient
from internal.rag.hybrid import HybridStore
from internal.rag.reranker import LLMReranker
from internal.rag.rewriter import LLMRewriter


LOGGER = logging.getLogger(__name__)
BENCHMARK_PREFIX = "milo-bench:"
ES_INDEX_NAME = "rag_chunks"


class ConfigureRequest(BaseModel):
    embedding_dim: Optional[int] = Field(default=None, ge=1, le=65536)
    benchmark_token: SecretStr


class BenchmarkChunk(BaseModel):
    chunk_id: str = Field(..., min_length=1)
    document_id: str = Field(..., min_length=1)
    chunk_index: int = Field(..., ge=0)
    content: str = Field(..., min_length=1, max_length=4096)
    content_sha256: str = Field(..., min_length=64, max_length=64)
    parent_content: str = Field(default="", max_length=10000)


class ImportRequest(BaseModel):
    chunks: List[BenchmarkChunk] = Field(..., min_length=1, max_length=200)
    confirm_embedding_cost: bool = False
    embedding_workers: int = Field(default=5, ge=1, le=16)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    rewrite: bool = False
    rerank: bool = False


@dataclass
class BenchmarkRuntime:
    cfg: APIConfig
    inf: Infrastructure
    llm: LLMClient


def benchmark_key(chunk_id: str) -> str:
    return f"{BENCHMARK_PREFIX}{chunk_id}"


def chunk_id_from_key(value: str) -> Optional[str]:
    if not value.startswith(BENCHMARK_PREFIX):
        return None
    return value[len(BENCHMARK_PREFIX) :]


def configure_runtime_config(
    base_cfg: APIConfig,
    request: ConfigureRequest,
    *,
    running_in_container: bool,
) -> APIConfig:
    cfg = copy.deepcopy(base_cfg)
    if request.embedding_dim is not None:
        cfg.rag_milvus_dim = request.embedding_dim
    cfg.temperature = 0.0

    if running_in_container:
        cfg.pg_host = "postgres"
        cfg.pg_port = 5432
        cfg.milvus_host = "milvus"
        cfg.milvus_port = 19530
        cfg.es_addresses = ["http://elasticsearch:9200"]
        cfg.es_username = ""
        cfg.es_password = ""
        cfg.kafka_brokers = ["kafka:19092"]

    # Published Milo-bench retrieval protocol.  Rewrite and rerank are chosen
    # per request so baseline/enhanced runs can share the same indexed corpus.
    cfg.chunk_size = 400
    cfg.chunk_overlap = 80
    cfg.top_k = 10
    cfg.rrf_constant_k = 60
    cfg.semantic_weight = 0.7
    cfg.enable_hybrid_search = True
    cfg.rag_rewrite_enabled = False
    cfg.rag_rerank_enabled = False
    cfg.kg_enabled = False
    cfg.neo4j_uri = ""
    cfg.sandbox_enabled = False
    return cfg


def _embedding_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            return False
    return isinstance(value, list) and bool(value)


def _as_embedding(value: Any) -> List[float]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list):
        return []
    return [float(item) for item in value]


class BenchmarkController:
    def __init__(self, base_cfg: APIConfig):
        self.base_cfg = base_cfg
        self._runtime: Optional[BenchmarkRuntime] = None
        self._token_hash = ""
        self._lock = threading.RLock()

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def authorize(self, token: Optional[str]) -> None:
        with self._lock:
            expected = self._token_hash
        if not expected or not token or not hmac.compare_digest(expected, self._hash_token(token)):
            raise HTTPException(status_code=403, detail="invalid benchmark token")

    def configure(
        self,
        request: ConfigureRequest,
        current_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        new_token = request.benchmark_token.get_secret_value()
        if len(new_token) < 16:
            raise HTTPException(status_code=400, detail="benchmark token must be at least 16 characters")
        with self._lock:
            already_configured = self._runtime is not None
        if already_configured:
            self.authorize(current_token)

        if not self.base_cfg.is_real_embedding():
            raise HTTPException(
                status_code=400,
                detail=(
                    "embedding API is not configured in config/config.local.yaml"
                ),
            )

        cfg = configure_runtime_config(
            self.base_cfg,
            request,
            running_in_container=os.path.exists("/.dockerenv"),
        )
        inf = Infrastructure(cfg)
        missing = [
            name
            for name in ("postgresql", "elasticsearch", "milvus")
            if getattr(inf.ready, name, "disconnected") != "connected"
        ]
        if missing:
            inf.close()
            raise HTTPException(
                status_code=503,
                detail=f"benchmark storage is not ready: {', '.join(missing)}",
            )
        runtime = BenchmarkRuntime(cfg=cfg, inf=inf, llm=LLMClient(cfg))
        with self._lock:
            previous = self._runtime
            self._runtime = runtime
            self._token_hash = self._hash_token(new_token)
        if previous is not None:
            previous.inf.close()
        return {
            "configured": True,
            "embedding_model": cfg.embedding_model,
            "llm_model": cfg.llm_model if cfg.is_real_llm() else "",
            "embedding_dim": cfg.rag_milvus_dim,
            "storage": {
                "postgresql": inf.ready.postgresql,
                "elasticsearch": inf.ready.elasticsearch,
                "milvus": inf.ready.milvus,
            },
        }

    def close(self) -> None:
        with self._lock:
            runtime = self._runtime
            self._runtime = None
            self._token_hash = ""
        if runtime is not None:
            runtime.inf.close()

    def _get_runtime(self) -> BenchmarkRuntime:
        with self._lock:
            runtime = self._runtime
        if runtime is None:
            raise HTTPException(
                status_code=409,
                detail="benchmark is not configured; call /api/benchmark/configure first",
            )
        return runtime

    @staticmethod
    def _pg_counts(runtime: BenchmarkRuntime) -> tuple[int, int, int]:
        if runtime.inf._pg is None:
            return 0, 0, 0
        with runtime.inf._pg.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*), "
                "COUNT(*) FILTER (WHERE doc_hash LIKE %s), "
                "COUNT(*) FILTER (WHERE doc_hash LIKE %s AND embedding IS NOT NULL "
                "AND embedding <> '[]'::jsonb) FROM rag_chunks",
                (f"{BENCHMARK_PREFIX}%", f"{BENCHMARK_PREFIX}%"),
            )
            row = cursor.fetchone() or (0, 0, 0)
        return int(row[0]), int(row[1]), int(row[2])

    def _require_isolated_corpus(self, runtime: BenchmarkRuntime) -> None:
        total, benchmark, _embeddings = self._pg_counts(runtime)
        if total != benchmark:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"rag_chunks contains {total - benchmark} non-benchmark rows; "
                    "use an empty AGI-Mira data volume for a valid benchmark"
                ),
            )

    @staticmethod
    def _existing(
        runtime: BenchmarkRuntime, keys: Sequence[str]
    ) -> Dict[str, Dict[str, Any]]:
        if not keys or runtime.inf._pg is None:
            return {}
        with runtime.inf._pg.cursor() as cursor:
            cursor.execute(
                "SELECT id, doc_hash, content, COALESCE(parent_content, ''), embedding "
                "FROM rag_chunks WHERE chunk_idx = 0 AND doc_hash = ANY(%s)",
                (list(keys),),
            )
            rows = cursor.fetchall()
        return {
            str(row[1]): {
                "pg_id": int(row[0]),
                "content": row[2] or "",
                "parent_content": row[3] or "",
                "embedding": row[4],
            }
            for row in rows
        }

    @staticmethod
    def _embed_many(
        runtime: BenchmarkRuntime,
        chunks: Sequence[BenchmarkChunk],
        workers: int,
    ) -> Dict[str, List[float]]:
        def embed_one(chunk: BenchmarkChunk) -> tuple[str, List[float]]:
            embedding = _as_embedding(runtime.llm.embed(chunk.content))
            if len(embedding) != int(runtime.cfg.rag_milvus_dim):
                raise RuntimeError(
                    f"chunk {chunk.chunk_id} embedding dim={len(embedding)}, "
                    f"configured dim={runtime.cfg.rag_milvus_dim}"
                )
            return benchmark_key(chunk.chunk_id), embedding

        if workers == 1:
            return dict(embed_one(chunk) for chunk in chunks)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="milo-embed") as pool:
            return dict(pool.map(embed_one, chunks))

    def import_chunks(self, request: ImportRequest) -> Dict[str, Any]:
        runtime = self._get_runtime()
        self._require_isolated_corpus(runtime)

        seen: set[str] = set()
        for chunk in request.chunks:
            if chunk.chunk_id in seen:
                raise HTTPException(status_code=400, detail=f"duplicate chunk_id: {chunk.chunk_id}")
            seen.add(chunk.chunk_id)
            actual = hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()
            if actual != chunk.content_sha256.lower():
                raise HTTPException(
                    status_code=400,
                    detail=f"content SHA-256 mismatch: {chunk.chunk_id}",
                )

        keys = [benchmark_key(chunk.chunk_id) for chunk in request.chunks]
        existing = self._existing(runtime, keys)
        to_embed = [
            chunk
            for chunk in request.chunks
            if benchmark_key(chunk.chunk_id) not in existing
            or existing[benchmark_key(chunk.chunk_id)]["content"] != chunk.content
            or not _embedding_present(existing[benchmark_key(chunk.chunk_id)]["embedding"])
        ]
        if to_embed and not request.confirm_embedding_cost:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"batch requires {len(to_embed)} embedding API calls; "
                    "set confirm_embedding_cost=true"
                ),
            )

        try:
            fresh_embeddings = self._embed_many(runtime, to_embed, request.embedding_workers)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"embedding failed: {exc}") from exc

        rows: List[Dict[str, Any]] = []
        for chunk in request.chunks:
            key = benchmark_key(chunk.chunk_id)
            embedding = fresh_embeddings.get(key)
            if embedding is None:
                embedding = _as_embedding(existing[key]["embedding"])
            pg_id = runtime.inf.repo.ragchunk.save_pg_with_parent(
                key,
                0,
                chunk.content,
                chunk.parent_content,
                json.dumps(embedding),
            )
            if pg_id <= 0:
                raise HTTPException(status_code=500, detail=f"PostgreSQL upsert failed: {chunk.chunk_id}")
            rows.append(
                {
                    "pg_id": pg_id,
                    "key": key,
                    "content": chunk.content,
                    "embedding": embedding,
                }
            )

        self._bulk_index_es(runtime, rows)
        self._bulk_upsert_milvus(runtime, rows)
        return {
            "received": len(request.chunks),
            "embedded": len(to_embed),
            "reused_embeddings": len(request.chunks) - len(to_embed),
            "first_chunk_id": request.chunks[0].chunk_id,
            "last_chunk_id": request.chunks[-1].chunk_id,
        }

    @staticmethod
    def _bulk_index_es(runtime: BenchmarkRuntime, rows: Sequence[Dict[str, Any]]) -> None:
        try:
            from elasticsearch.helpers import bulk

            actions = [
                {
                    "_op_type": "index",
                    "_index": ES_INDEX_NAME,
                    "_id": row["pg_id"],
                    "_source": {
                        "pg_id": row["pg_id"],
                        "content": row["content"],
                        "doc_hash": row["key"],
                        "chunk_idx": 0,
                    },
                }
                for row in rows
            ]
            bulk(runtime.inf._es, actions, refresh=False, raise_on_error=True)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Elasticsearch bulk index failed: {exc}") from exc

    @staticmethod
    def _bulk_upsert_milvus(runtime: BenchmarkRuntime, rows: Sequence[Dict[str, Any]]) -> None:
        data = [
            {
                "pg_id": int(row["pg_id"]),
                "content": row["content"],
                "embedding": row["embedding"],
            }
            for row in rows
        ]
        try:
            runtime.inf._milvus.upsert(collection_name=RAG_COLLECTION, data=data)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Milvus upsert failed: {exc}") from exc

    @staticmethod
    def _milvus_entity_count(runtime: BenchmarkRuntime) -> int:
        try:
            rows = runtime.inf._milvus.query(
                collection_name=RAG_COLLECTION,
                filter="",
                output_fields=["count(*)"],
                timeout=30.0,
            )
            if rows and rows[0].get("count(*)") is not None:
                return int(rows[0]["count(*)"])
        except Exception as exc:
            LOGGER.warning("Milvus count(*) query failed; using row_count: %s", exc)
        stats = runtime.inf._milvus.get_collection_stats(RAG_COLLECTION)
        return int(stats.get("row_count", 0))

    def finalize(self) -> Dict[str, Any]:
        runtime = self._get_runtime()
        self._require_isolated_corpus(runtime)
        try:
            runtime.inf._es.indices.refresh(index=ES_INDEX_NAME)
            flush = getattr(runtime.inf._milvus, "flush", None)
            if callable(flush):
                flush(collection_name=RAG_COLLECTION)
            runtime.inf._milvus.load_collection(collection_name=RAG_COLLECTION)

            _total, expected, _embeddings = self._pg_counts(runtime)
            deadline = time.monotonic() + 90.0
            while True:
                actual = self._milvus_entity_count(runtime)
                if actual >= expected:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Milvus visibility timeout: count={actual}, expected={expected}"
                    )
                time.sleep(1.0)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"benchmark finalize failed: {exc}") from exc
        return self.status()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            runtime = self._runtime
        if runtime is None:
            return {"configured": False}

        total, postgres, embeddings = self._pg_counts(runtime)
        elasticsearch = 0
        try:
            response = runtime.inf._es.count(
                index=ES_INDEX_NAME,
                body={"query": {"prefix": {"doc_hash": BENCHMARK_PREFIX}}},
            )
            elasticsearch = int(response.get("count", 0))
        except Exception:
            elasticsearch = -1

        milvus = 0
        try:
            milvus = self._milvus_entity_count(runtime)
        except Exception:
            milvus = -1

        return {
            "configured": True,
            "counts": {
                "postgres": postgres,
                "postgres_embeddings": embeddings,
                "elasticsearch": elasticsearch,
                "milvus": milvus,
                "non_benchmark_postgres": total - postgres,
            },
            "models": {
                "embedding": runtime.cfg.embedding_model,
                "llm": runtime.cfg.llm_model if runtime.cfg.is_real_llm() else "",
                "embedding_dim": runtime.cfg.rag_milvus_dim,
                "wire_api": "chat_completions" if runtime.cfg.is_real_llm() else "unused",
            },
        }

    @staticmethod
    def _strict_generate(runtime: BenchmarkRuntime, system: str, user: str) -> str:
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        response = requests.post(
            runtime.cfg.llm_api_url,
            headers={
                "Authorization": f"Bearer {runtime.cfg.llm_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": runtime.cfg.llm_model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 4096,
                "stream": False,
            },
            timeout=(10, 180),
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Chat Completions returned HTTP {response.status_code}: {response.text}"
            )
        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError("Chat Completions returned non-JSON data") from exc
        error = data.get("error") if isinstance(data, dict) else None
        if error:
            message = error.get("message") if isinstance(error, dict) else str(error)
            raise RuntimeError(f"Chat Completions API error: {message}")
        choices = data.get("choices", []) if isinstance(data, dict) else []
        if not choices:
            raise RuntimeError("Chat Completions returned no choices")
        choice = choices[0] or {}
        if choice.get("finish_reason") == "length":
            raise RuntimeError(
                "Chat Completions output hit max_tokens=4096 before JSON completed"
            )
        content = (choice.get("message") or {}).get("content") or ""
        if not content:
            raise RuntimeError("Chat Completions returned empty message content")
        return str(content)

    @staticmethod
    def _metadata_for_ids(
        runtime: BenchmarkRuntime, ids: Sequence[int]
    ) -> Dict[int, Dict[str, Any]]:
        if not ids:
            return {}
        with runtime.inf._pg.cursor() as cursor:
            cursor.execute(
                "SELECT id, doc_hash, content FROM rag_chunks WHERE id = ANY(%s)",
                (list(ids),),
            )
            rows = cursor.fetchall()
        output: Dict[int, Dict[str, Any]] = {}
        for pg_id, key, content in rows:
            chunk_id = chunk_id_from_key(str(key))
            if chunk_id is not None:
                output[int(pg_id)] = {"chunk_id": chunk_id, "content": content or ""}
        return output

    def search(self, request: SearchRequest) -> Dict[str, Any]:
        runtime = self._get_runtime()
        self._require_isolated_corpus(runtime)
        if (request.rewrite or request.rerank) and not runtime.cfg.is_real_llm():
            raise HTTPException(status_code=400, detail="LLM API is required for rewrite/rerank")
        if request.rewrite or request.rerank:
            if not str(runtime.cfg.llm_api_url or "").rstrip("/").endswith(
                "/chat/completions"
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "AGI-Mira benchmark requires a complete Chat Completions URL "
                        "ending in /chat/completions"
                    ),
                )

        generate = lambda system, user: self._strict_generate(runtime, system, user)
        queries = [request.query]
        try:
            if request.rewrite:
                queries = LLMRewriter(
                    generate,
                    runtime.cfg.rag_rewrite_num_queries,
                    strict=True,
                ).rewrite(request.query, [])
                if len(queries) <= 1:
                    raise RuntimeError(
                        "query rewrite was requested but not applied; check LLM URL, "
                        "model, API key, and response format"
                    )

            store = HybridStore(runtime.cfg, runtime.inf, embed_fn=runtime.llm.embed)
            if request.rerank:
                store.set_reranker(
                    LLMReranker(
                        generate,
                        runtime.cfg.rag_rerank_preview_len,
                        strict=True,
                    )
                )
            hits = store.search_multi(queries, request.top_k)
            if request.rerank and not any(
                str(hit.source).endswith("+rerank") for hit in hits
            ):
                raise RuntimeError(
                    "rerank was requested but not applied; check LLM URL, model, "
                    "API key, and response format"
                )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"retrieval failed: {exc}") from exc

        metadata = self._metadata_for_ids(runtime, [int(hit.pg_id) for hit in hits])
        results = []
        for hit in hits:
            row = metadata.get(int(hit.pg_id))
            if row is None:
                continue
            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "content": row["content"],
                    "score": float(hit.score),
                    "source": hit.source,
                }
            )
        return {
            "query": request.query,
            "rewritten_queries": queries,
            "rewrite": request.rewrite,
            "rerank": request.rerank,
            "wire_api": "chat_completions" if request.rewrite or request.rerank else "unused",
            "results": results,
        }


def setup_benchmark_routes(app: FastAPI, cfg: APIConfig) -> BenchmarkController:
    controller = BenchmarkController(cfg)
    first_benchmark_route = len(app.router.routes)

    @app.post("/api/benchmark/configure")
    def benchmark_configure(
        request: ConfigureRequest,
        x_benchmark_token: Optional[str] = Header(default=None),
    ):
        return controller.configure(request, current_token=x_benchmark_token)

    @app.get("/api/benchmark/status")
    def benchmark_status():
        return controller.status()

    @app.post("/api/benchmark/import")
    def benchmark_import(
        request: ImportRequest,
        x_benchmark_token: Optional[str] = Header(default=None),
    ):
        controller.authorize(x_benchmark_token)
        return controller.import_chunks(request)

    @app.post("/api/benchmark/finalize")
    def benchmark_finalize(x_benchmark_token: Optional[str] = Header(default=None)):
        controller.authorize(x_benchmark_token)
        return controller.finalize()

    @app.post("/api/benchmark/search")
    def benchmark_search(
        request: SearchRequest,
        x_benchmark_token: Optional[str] = Header(default=None),
    ):
        controller.authorize(x_benchmark_token)
        return controller.search(request)

    app.router.on_shutdown.append(controller.close)

    # setup_routes mounts the frontend at "/" before returning.  Starlette
    # resolves routes in registration order, so API routes appended after that
    # catch-all mount would be hidden.  Move only the newly added benchmark
    # routes immediately before the root mount.
    new_routes = list(app.router.routes[first_benchmark_route:])
    original_routes = list(app.router.routes[:first_benchmark_route])
    mount_index = next(
        (
            index
            for index, route in enumerate(original_routes)
            if isinstance(route, Mount) and getattr(route, "path", None) == ""
        ),
        len(original_routes),
    )
    app.router.routes[:] = (
        original_routes[:mount_index] + new_routes + original_routes[mount_index:]
    )

    return controller
