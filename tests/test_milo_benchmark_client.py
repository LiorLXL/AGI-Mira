import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.evaluate_milo import (
    benchmark_token,
    configure,
    evaluate,
    evaluate_one,
    finalize,
    hit_rate_at_k,
    import_payload,
    ndcg_at_k,
    parse_args,
    recall_at_k,
    reciprocal_rank,
)


class MiloBenchmarkClientTest(unittest.TestCase):
    def test_configure_allows_dimension_to_come_from_local_yaml(self):
        args = parse_args(["configure"])

        self.assertIsNone(args.embedding_dim)

    def test_enhanced_evaluate_defaults_to_one_retry(self):
        args = parse_args(["evaluate", "--rewrite", "--rerank"])

        self.assertEqual(args.api_retries, 1)
        self.assertFalse(args.resume)

    @patch("benchmark.evaluate_milo.request_json")
    def test_finalize_does_not_import_or_embed(self, request_json):
        request_json.return_value = {"counts": {"milvus": 23493}}
        args = argparse.Namespace(
            base_url="http://127.0.0.1:8090",
            request_timeout=120.0,
            api_retries=2,
        )

        result = finalize(args)

        self.assertEqual(result["counts"]["milvus"], 23493)
        self.assertEqual(request_json.call_args.args[0], "POST")
        self.assertTrue(request_json.call_args.args[1].endswith("/api/benchmark/finalize"))
        self.assertIsNone(request_json.call_args.args[2])

    def test_metrics_match_ranked_and_graded_qrels(self):
        ranked = ["relevant-high", "noise", "relevant-low"]
        relevant = {"relevant-high", "relevant-low"}
        relevance = {"relevant-high": 3, "relevant-low": 1}

        self.assertEqual(recall_at_k(ranked, relevant, 3), 1.0)
        self.assertEqual(reciprocal_rank(ranked, relevant, 10), 1.0)
        self.assertEqual(hit_rate_at_k(ranked, relevant, 1), 1.0)
        self.assertAlmostEqual(ndcg_at_k(ranked, relevance, 3), 0.982842, places=6)

    def test_import_payload_preserves_fixed_chunk_identity(self):
        rows = [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "chunk_index": 7,
                "content": "fixed content",
                "content_sha256": "a" * 64,
                "parent_content": "parent",
                "ignored": True,
            }
        ]

        payload = import_payload(rows)

        self.assertEqual(
            payload,
            [
                {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "chunk_index": 7,
                    "content": "fixed content",
                    "content_sha256": "a" * 64,
                    "parent_content": "parent",
                }
            ],
        )

    @patch("benchmark.evaluate_milo.benchmark_token", return_value="benchmark-token-123456")
    @patch("benchmark.evaluate_milo.request_json")
    def test_configure_sends_only_local_token_and_optional_dimension(
        self, request_json, _benchmark_token
    ):
        request_json.return_value = {"configured": True}
        args = argparse.Namespace(
            base_url="http://127.0.0.1:8090",
            embedding_dim=2048,
        )

        result = configure(args)

        self.assertEqual(result, {"configured": True})
        payload = request_json.call_args.args[2]
        self.assertEqual(
            payload,
            {
                "benchmark_token": "benchmark-token-123456",
                "embedding_dim": 2048,
            },
        )

    def test_benchmark_token_is_created_once_and_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / ".benchmark-token"
            with patch("benchmark.evaluate_milo.TOKEN_PATH", token_path), patch.dict(
                "os.environ", {}, clear=True
            ):
                first = benchmark_token()
                second = benchmark_token()

        self.assertGreaterEqual(len(first), 16)
        self.assertEqual(first, second)

    @patch("benchmark.evaluate_milo.status")
    @patch("benchmark.evaluate_milo.evaluate_one")
    def test_evaluate_checkpoints_and_resumes_after_failure(self, evaluate_one, get_status):
        get_status.return_value = {
            "configured": True,
            "models": {"llm": "model", "embedding": "embedding", "wire_api": "responses"},
        }

        def detail(query_id):
            metrics = {
                "recall@3": 1.0,
                "recall@5": 1.0,
                "recall@10": 1.0,
                "mrr@10": 1.0,
                "ndcg@3": 1.0,
                "ndcg@5": 1.0,
                "ndcg@10": 1.0,
                "hit_rate@1": 1.0,
                "hit_rate@3": 1.0,
                "hit_rate@5": 1.0,
                "hit_rate@10": 1.0,
            }
            return {
                "query_id": query_id,
                "question": query_id,
                "wire_api": "responses",
                "rewritten_queries": [query_id, f"{query_id}-alt"],
                "ground_truth": [f"{query_id}-chunk"],
                "metrics": metrics,
                "latency_ms": 1.0,
                "results": [],
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.jsonl"
            gold_path.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "query_id": query_id,
                            "question": query_id,
                            "ground_truth": [f"{query_id}-chunk"],
                            "relevance": {f"{query_id}-chunk": 3},
                            "split": "dev",
                        }
                    )
                    for query_id in ("q1", "q2")
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "run"
            args = argparse.Namespace(
                base_url="http://127.0.0.1:8090",
                milo_root=root,
                gold=gold_path,
                query_limit=None,
                rewrite=True,
                rerank=True,
                evaluation_workers=1,
                api_retries=1,
                request_timeout=10.0,
                output_dir=output_dir,
                resume=False,
            )
            evaluate_one.side_effect = [detail("q1"), RuntimeError("503 exhausted")]

            with self.assertRaisesRegex(RuntimeError, "503 exhausted"):
                evaluate(args)

            checkpoint = (output_dir / "eval_details.jsonl").read_text(encoding="utf-8")
            self.assertEqual(len([line for line in checkpoint.splitlines() if line]), 1)

            args.resume = True
            evaluate_one.side_effect = [detail("q2")]
            summary = evaluate(args)

            self.assertEqual(summary["benchmark"]["queries"], 2)
            self.assertEqual(evaluate_one.call_count, 3)

    @patch("benchmark.evaluate_milo.request_json")
    def test_evaluate_one_scores_agi_mira_rankings(self, request_json):
        request_json.return_value = {
            "rewritten_queries": ["question"],
            "results": [
                {"chunk_id": "chunk-1", "content": "answer", "score": 1.0, "source": "hybrid"},
                {"chunk_id": "noise", "content": "noise", "score": 0.5, "source": "hybrid"},
            ],
        }
        item = {
            "query_id": "q1",
            "question": "question",
            "ground_truth": ["chunk-1"],
            "relevance": {"chunk-1": 3},
        }

        detail = evaluate_one(
            "http://127.0.0.1:8090",
            item,
            rewrite=False,
            rerank=False,
            api_retries=1,
            request_timeout=10,
        )

        self.assertEqual(detail["metrics"]["recall@10"], 1.0)
        self.assertEqual(detail["metrics"]["mrr@10"], 1.0)
        self.assertTrue(detail["results"][0]["relevant"])


if __name__ == "__main__":
    unittest.main()
