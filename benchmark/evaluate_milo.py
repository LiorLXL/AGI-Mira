#!/usr/bin/env python3
"""Drive AGI-Mira's benchmark HTTP API with Milo-bench data on Windows.

The script intentionally does not connect to PostgreSQL, Elasticsearch or
Milvus.  AGI-Mira is the only system under test; this client only materializes
Milo-bench data, imports immutable chunks over HTTP, asks retrieval questions,
and calculates the published metric set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import statistics
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:8090"
DEFAULT_MILO_ROOT = Path(__file__).resolve().parents[2] / "Milo-bench"
CORPUS_RELATIVE = Path("test_data/rag_eval_v2/chunks_400_80.jsonl")
GOLD_RELATIVE = Path("test_data/rag_eval_v2/benchmark_dev_120_qrels_ab_fill.jsonl")
METRIC_KS = (1, 3, 5, 10)
TOKEN_PATH = Path(__file__).with_name(".benchmark-token")


class BenchmarkClientError(RuntimeError):
    pass


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkClientError(f"invalid JSONL {path}:{number}: {exc}") from exc
            if not isinstance(row, dict):
                raise BenchmarkClientError(f"JSONL row is not an object: {path}:{number}")
            rows.append(row)
    return rows


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BenchmarkClientError(f"invalid JSONL {path}:{number}: {exc}") from exc
            if not isinstance(row, dict):
                raise BenchmarkClientError(f"JSONL row is not an object: {path}:{number}")
            yield row


def batched(rows: Iterable[Dict[str, Any]], size: int) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def ensure_corpus(milo_root: Path) -> Path:
    corpus = milo_root / CORPUS_RELATIVE
    if corpus.is_file():
        return corpus
    materializer = milo_root / "scripts/materialize_corpus.py"
    if not materializer.is_file():
        raise BenchmarkClientError(f"Milo-bench materializer not found: {materializer}")
    completed = subprocess.run(
        [sys.executable, str(materializer)],
        cwd=str(milo_root),
        check=False,
    )
    if completed.returncode != 0 or not corpus.is_file():
        raise BenchmarkClientError("failed to materialize Milo-bench corpus")
    return corpus


def endpoint(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def benchmark_token() -> str:
    """Load or create the local benchmark API token.

    The token is unrelated to provider credentials.  It only protects local
    billable benchmark endpoints and is stored in a Git-ignored file.
    """

    from_environment = os.environ.get("AGI_BENCHMARK_TOKEN", "").strip()
    if from_environment:
        if len(from_environment) < 16:
            raise BenchmarkClientError("AGI_BENCHMARK_TOKEN must be at least 16 characters")
        return from_environment
    if TOKEN_PATH.is_file():
        value = TOKEN_PATH.read_text(encoding="utf-8").strip()
        if len(value) >= 16:
            return value
    value = secrets.token_urlsafe(32)
    TOKEN_PATH.write_text(value + "\n", encoding="utf-8")
    try:
        TOKEN_PATH.chmod(0o600)
    except OSError:
        pass
    return value


def request_json(
    method: str,
    url: str,
    payload: Optional[Mapping[str, Any]] = None,
    *,
    timeout: float = 600.0,
    retries: int = 1,
) -> Dict[str, Any]:
    headers = {"X-Benchmark-Token": benchmark_token()}
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.request(
                method,
                url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", response.text)
                except Exception:
                    detail = response.text
                raise BenchmarkClientError(f"HTTP {response.status_code} {url}: {detail}")
            data = response.json()
            if not isinstance(data, dict):
                raise BenchmarkClientError(f"HTTP response is not an object: {url}")
            return data
        except (requests.RequestException, BenchmarkClientError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise BenchmarkClientError(str(last_error or f"request failed: {url}"))


def configure(args: argparse.Namespace) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"benchmark_token": benchmark_token()}
    if args.embedding_dim is not None:
        payload["embedding_dim"] = args.embedding_dim
    return request_json(
        "POST",
        endpoint(args.base_url, "/api/benchmark/configure"),
        payload,
        timeout=60,
    )


def status(base_url: str) -> Dict[str, Any]:
    return request_json(
        "GET",
        endpoint(base_url, "/api/benchmark/status"),
        timeout=30,
    )


def finalize(args: argparse.Namespace) -> Dict[str, Any]:
    return request_json(
        "POST",
        endpoint(args.base_url, "/api/benchmark/finalize"),
        None,
        timeout=args.request_timeout,
        retries=args.api_retries,
    )


def import_payload(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    required = {
        "chunk_id",
        "document_id",
        "chunk_index",
        "content",
        "content_sha256",
    }
    output: List[Dict[str, Any]] = []
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            raise BenchmarkClientError(f"corpus row missing fields: {missing}")
        output.append(
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "content_sha256": row["content_sha256"],
                "parent_content": row.get("parent_content", ""),
            }
        )
    return output


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.confirm_embedding_cost:
        raise BenchmarkClientError(
            "prepare can issue up to 23,493 billable embedding calls; "
            "rerun with --confirm-embedding-cost"
        )
    milo_root = args.milo_root.resolve()
    corpus = ensure_corpus(milo_root)
    imported = 0
    embedded = 0
    started = time.perf_counter()
    for batch in batched(iter_jsonl(corpus), args.batch_size):
        result = request_json(
            "POST",
            endpoint(args.base_url, "/api/benchmark/import"),
            {
                "chunks": import_payload(batch),
                "confirm_embedding_cost": True,
                "embedding_workers": args.embedding_workers,
            },
            timeout=args.request_timeout,
            retries=args.api_retries,
        )
        imported += int(result.get("received", 0))
        embedded += int(result.get("embedded", 0))
        elapsed = time.perf_counter() - started
        print(
            f"imported={imported} embedded_this_run={embedded} "
            f"elapsed={elapsed:.1f}s last={result.get('last_chunk_id', '')}",
            flush=True,
        )

    final = request_json(
        "POST",
        endpoint(args.base_url, "/api/benchmark/finalize"),
        timeout=args.request_timeout,
        retries=args.api_retries,
    )
    counts = final.get("counts", {})
    expected = imported
    for name in ("postgres", "postgres_embeddings", "elasticsearch", "milvus"):
        actual = int(counts.get(name, -1))
        if actual != expected:
            raise BenchmarkClientError(
                f"incomplete benchmark index: {name}={actual}, expected={expected}"
            )
    return final


def reciprocal_rank(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(ranked_ids[:k], 1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def hit_rate_at_k(ranked_ids: Sequence[str], relevant: set[str], k: int) -> float:
    return float(bool(set(ranked_ids[:k]) & relevant))


def ndcg_at_k(ranked_ids: Sequence[str], relevance: Mapping[str, float], k: int) -> float:
    def gain(value: float) -> float:
        return (2.0 ** value) - 1.0

    dcg = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:k], 1):
        dcg += gain(float(relevance.get(chunk_id, 0.0))) / math.log2(rank + 1)
    ideal = sorted((float(value) for value in relevance.values()), reverse=True)[:k]
    idcg = sum(gain(rel) / math.log2(rank + 1) for rank, rel in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve_evaluation_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    name = "agi_mira_enhanced" if args.rewrite or args.rerank else "agi_mira_baseline"
    return args.milo_root.resolve() / "results/runs" / name


def evaluate_one(
    base_url: str,
    item: Mapping[str, Any],
    *,
    rewrite: bool,
    rerank: bool,
    api_retries: int,
    request_timeout: float,
) -> Dict[str, Any]:
    started = time.perf_counter()
    response = request_json(
        "POST",
        endpoint(base_url, "/api/benchmark/search"),
        {
            "query": item["question"],
            "top_k": 10,
            "rewrite": rewrite,
            "rerank": rerank,
        },
        timeout=request_timeout,
        retries=api_retries,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    results = list(response.get("results", []))
    ranked_ids = [str(result["chunk_id"]) for result in results]
    relevant = {str(value) for value in item["ground_truth"]}
    relevance = {
        str(key): float(value)
        for key, value in (item.get("relevance", {}) or {}).items()
    }
    metrics = {
        "recall@3": recall_at_k(ranked_ids, relevant, 3),
        "recall@5": recall_at_k(ranked_ids, relevant, 5),
        "recall@10": recall_at_k(ranked_ids, relevant, 10),
        "mrr@10": reciprocal_rank(ranked_ids, relevant, 10),
        "ndcg@3": ndcg_at_k(ranked_ids, relevance, 3),
        "ndcg@5": ndcg_at_k(ranked_ids, relevance, 5),
        "ndcg@10": ndcg_at_k(ranked_ids, relevance, 10),
    }
    for k in METRIC_KS:
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(ranked_ids, relevant, k)
    return {
        "query_id": item["query_id"],
        "question": item["question"],
        "wire_api": response.get("wire_api", "unknown"),
        "rewritten_queries": response.get("rewritten_queries", [item["question"]]),
        "ground_truth": list(item["ground_truth"]),
        "metrics": metrics,
        "latency_ms": round(latency_ms, 3),
        "results": [
            {
                "rank": rank,
                "chunk_id": result["chunk_id"],
                "score": float(result.get("score", 0.0)),
                "source": result.get("source", "unknown"),
                "relevant": result["chunk_id"] in relevant,
                "relevance_grade": relevance.get(result["chunk_id"], 0.0),
                "content_preview": str(result.get("content", ""))[:160],
            }
            for rank, result in enumerate(results, 1)
        ],
    }


def render_report(summary: Mapping[str, Any]) -> str:
    metrics = summary["metrics"]
    latency = summary["latency_ms"]
    pipeline = summary["pipeline"]
    lines = [
        "# AGI-Mira / Milo-bench Retrieval Evaluation",
        "",
        "## Pipeline",
        "",
        f"- Queries: `{summary['benchmark']['queries']}`",
        f"- Query rewrite: `{pipeline['query_rewrite']}`",
        f"- LLM rerank: `{pipeline['rerank']}`",
        "- Retrieval: AGI-Mira Elasticsearch + Milvus + weighted RRF",
        "- Fixed chunks: `400/80`",
        "- TopK: `10`",
        "- Semantic weight: `0.7`",
        "- RRF constant: `60`",
        "- Neo4j: `false`",
        "",
        "## Metrics",
        "",
        "| Metric | Score |",
        "|---|---:|",
    ]
    for name in (
        "recall@3",
        "recall@5",
        "recall@10",
        "mrr@10",
        "ndcg@3",
        "ndcg@5",
        "ndcg@10",
        "hit_rate@1",
        "hit_rate@3",
        "hit_rate@5",
        "hit_rate@10",
    ):
        lines.append(f"| {name} | {float(metrics[name]):.6f} |")
    lines.extend(
        [
            "",
            "## Latency",
            "",
            "| Statistic | Milliseconds |",
            "|---|---:|",
            f"| Mean | {float(latency['mean']):.3f} |",
            f"| P50 | {float(latency['p50']):.3f} |",
            f"| P95 | {float(latency['p95']):.3f} |",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    gold_path = args.gold or (args.milo_root.resolve() / GOLD_RELATIVE)
    gold = read_jsonl(gold_path)
    if args.query_limit is not None:
        gold = gold[: args.query_limit]
    if not gold:
        raise BenchmarkClientError("no evaluation questions selected")

    output_dir = resolve_evaluation_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "eval_details.jsonl"
    run_config_path = output_dir / "run_config.json"
    failure_path = output_dir / "last_failure.json"

    runtime_status = status(args.base_url)
    if not runtime_status.get("configured"):
        raise BenchmarkClientError("benchmark is not configured; run configure first")
    run_config = {
        "version": 1,
        "base_url": args.base_url.rstrip("/"),
        "gold": str(gold_path.resolve()),
        "gold_sha256": sha256_file(gold_path),
        "query_ids": [str(item["query_id"]) for item in gold],
        "rewrite": bool(args.rewrite),
        "rerank": bool(args.rerank),
        "top_k": 10,
        "models": runtime_status.get("models", {}),
    }

    if args.resume:
        if not run_config_path.is_file():
            raise BenchmarkClientError(
                f"cannot resume without {run_config_path}; use a new output directory"
            )
        previous_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        if previous_config != run_config:
            raise BenchmarkClientError(
                "cannot resume: dataset, model, protocol, or pipeline configuration changed"
            )
    else:
        write_json(run_config_path, run_config)
        details_path.write_text("", encoding="utf-8")
        failure_path.unlink(missing_ok=True)

    query_index = {str(item["query_id"]): index for index, item in enumerate(gold)}
    outcomes: List[Optional[Dict[str, Any]]] = [None] * len(gold)
    if args.resume and details_path.is_file():
        for row in read_jsonl(details_path):
            query_id = str(row.get("query_id", ""))
            if query_id not in query_index:
                raise BenchmarkClientError(f"resume file contains unknown query_id: {query_id}")
            index = query_index[query_id]
            if outcomes[index] is not None:
                raise BenchmarkClientError(f"resume file contains duplicate query_id: {query_id}")
            outcomes[index] = row

    completed_before_start = sum(outcome is not None for outcome in outcomes)
    if completed_before_start:
        print(
            f"resuming: {completed_before_start}/{len(gold)} queries already complete",
            flush=True,
        )

    def run(index: int, item: Mapping[str, Any]) -> tuple[int, Dict[str, Any]]:
        outcome = evaluate_one(
            args.base_url,
            item,
            rewrite=args.rewrite,
            rerank=args.rerank,
            api_retries=args.api_retries,
            request_timeout=args.request_timeout,
        )
        outcome["evaluation_workers"] = args.evaluation_workers
        return index, outcome

    def record(index: int, outcome: Dict[str, Any]) -> None:
        outcomes[index] = outcome
        with details_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(outcome, ensure_ascii=False) + "\n")
            handle.flush()
        completed = sum(value is not None for value in outcomes)
        print(f"evaluated {completed}/{len(gold)} {outcome['query_id']}", flush=True)

    def record_failure(index: int, exc: BaseException) -> None:
        write_json(
            failure_path,
            {
                "query_id": gold[index]["query_id"],
                "error": str(exc),
                "completed": sum(value is not None for value in outcomes),
                "total": len(gold),
                "resume_command_hint": "rerun the same command with --resume",
            },
        )

    pending_items = [
        (index, item)
        for index, item in enumerate(gold)
        if outcomes[index] is None
    ]
    if args.evaluation_workers == 1:
        for index, item in pending_items:
            try:
                _, outcome = run(index, item)
            except BaseException as exc:
                record_failure(index, exc)
                raise
            record(index, outcome)
    else:
        executor = ThreadPoolExecutor(
            max_workers=args.evaluation_workers,
            thread_name_prefix="agi-mira-eval",
        )
        pending_iterator = iter(pending_items)
        in_flight: Dict[Future, int] = {}

        def submit_next() -> bool:
            try:
                index, item = next(pending_iterator)
            except StopIteration:
                return False
            in_flight[executor.submit(run, index, item)] = index
            return True

        for _ in range(min(args.evaluation_workers, len(pending_items))):
            submit_next()
        try:
            while in_flight:
                done, _not_done = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    expected_index = in_flight.pop(future)
                    try:
                        index, outcome = future.result()
                    except BaseException as exc:
                        record_failure(expected_index, exc)
                        raise
                    record(index, outcome)
                    submit_next()
        except BaseException:
            for future in in_flight:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

    details = [outcome for outcome in outcomes if outcome is not None]
    if len(details) != len(gold):
        raise BenchmarkClientError(
            f"evaluation incomplete: {len(details)}/{len(gold)}; rerun with --resume"
        )
    metric_names = list(details[0]["metrics"])
    metrics = {
        name: round(statistics.fmean(row["metrics"][name] for row in details), 6)
        for name in metric_names
    }
    latencies = [float(row["latency_ms"]) for row in details]
    worker_counts = sorted(
        {int(row.get("evaluation_workers", args.evaluation_workers)) for row in details}
    )
    summary = {
        "benchmark": {
            "queries": len(details),
            "gold": str(gold_path),
            "split": sorted({str(item.get("split", "unknown")) for item in gold}),
        },
        "pipeline": {
            "system": "AGI-Mira HTTP benchmark API",
            "fixed_chunks": True,
            "query_rewrite": args.rewrite,
            "rerank": args.rerank,
            "top_k": 10,
            "semantic_weight": 0.7,
            "rrf_constant_k": 60,
            "neo4j": False,
            "evaluation_workers": worker_counts[0] if len(worker_counts) == 1 else worker_counts,
            "latency_comparable": len(worker_counts) == 1,
        },
        "metrics": metrics,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3),
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
        },
    }

    with details_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output_dir / "eval_summary.json", summary)
    (output_dir / "eval_report.md").write_text(render_report(summary), encoding="utf-8")
    failure_path.unlink(missing_ok=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote results to {output_dir}")
    return summary


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)


def add_request_args(parser: argparse.ArgumentParser, *, api_retries: int = 3) -> None:
    parser.add_argument("--api-retries", type=int, default=api_retries)
    parser.add_argument("--request-timeout", type=float, default=600.0)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    configure_parser = subparsers.add_parser(
        "configure",
        help="initialize benchmark runtime from config/config.local.yaml",
    )
    add_connection_args(configure_parser)
    configure_parser.add_argument(
        "--embedding-dim",
        type=int,
        help="override config.local.yaml rag.rag_milvus_dim",
    )

    status_parser = subparsers.add_parser("status", help="show benchmark index status")
    add_connection_args(status_parser)

    finalize_parser = subparsers.add_parser(
        "finalize",
        help="refresh indexes and verify counts without importing or embedding",
    )
    add_connection_args(finalize_parser)
    add_request_args(finalize_parser)

    prepare_parser = subparsers.add_parser("prepare", help="import fixed Milo chunks")
    add_connection_args(prepare_parser)
    add_request_args(prepare_parser)
    prepare_parser.add_argument("--milo-root", type=Path, default=DEFAULT_MILO_ROOT)
    prepare_parser.add_argument("--batch-size", type=int, default=50)
    prepare_parser.add_argument("--embedding-workers", type=int, default=5)
    prepare_parser.add_argument("--confirm-embedding-cost", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate Dev qrels over HTTP")
    add_connection_args(evaluate_parser)
    add_request_args(evaluate_parser, api_retries=1)
    evaluate_parser.add_argument("--milo-root", type=Path, default=DEFAULT_MILO_ROOT)
    evaluate_parser.add_argument("--gold", type=Path)
    evaluate_parser.add_argument("--query-limit", type=int)
    evaluate_parser.add_argument("--evaluation-workers", type=int, default=1)
    evaluate_parser.add_argument("--rewrite", action="store_true")
    evaluate_parser.add_argument("--rerank", action="store_true")
    evaluate_parser.add_argument(
        "--resume",
        action="store_true",
        help="resume completed queries from the output directory checkpoint",
    )
    evaluate_parser.add_argument("--output-dir", type=Path)

    args = parser.parse_args(argv)
    for name in ("api_retries", "batch_size", "embedding_workers", "evaluation_workers"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if getattr(args, "query_limit", None) is not None and args.query_limit <= 0:
        parser.error("--query-limit must be positive")
    embedding_dim = getattr(args, "embedding_dim", None)
    if embedding_dim is not None and embedding_dim <= 0:
        parser.error("--embedding-dim must be positive")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "configure":
            print(json.dumps(configure(args), ensure_ascii=False, indent=2))
        elif args.command == "status":
            print(json.dumps(status(args.base_url), ensure_ascii=False, indent=2))
        elif args.command == "finalize":
            print(json.dumps(finalize(args), ensure_ascii=False, indent=2))
        elif args.command == "prepare":
            print(json.dumps(prepare(args), ensure_ascii=False, indent=2))
        elif args.command == "evaluate":
            evaluate(args)
        else:
            raise BenchmarkClientError(f"unsupported command: {args.command}")
        return 0
    except BenchmarkClientError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
