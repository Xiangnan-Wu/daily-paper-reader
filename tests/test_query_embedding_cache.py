import importlib.util
import pathlib
import sys
import tempfile
import unittest

import numpy as np
import yaml


def _load_module(module_name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


class QueryEmbeddingCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = pathlib.Path(__file__).resolve().parents[1]
        src_dir = root / "src"
        sys.path.insert(0, str(src_dir))
        cls.mod = _load_module(
            "embedding_cache_mod",
            src_dir / "2.2.retrieval_papers_embedding.py",
        )

    def test_build_query_embedding_hash_is_stable(self):
        h1 = self.mod.build_query_embedding_hash("BAAI/bge-small-en-v1.5", "symbolic regression")
        h2 = self.mod.build_query_embedding_hash("BAAI/bge-small-en-v1.5", "symbolic regression")
        h3 = self.mod.build_query_embedding_hash("BAAI/bge-small-en-v1.5", "equation discovery")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)

    def test_hydrate_query_embeddings_uses_cache_and_only_encodes_misses(self):
        cfg = {
            "subscriptions": {
                "embedding_cache": {
                    "version": 1,
                    "query_vectors": {},
                }
            }
        }
        cached_hash = self.mod.build_query_embedding_hash("BAAI/bge-small-en-v1.5", "cached query")
        cfg["subscriptions"]["embedding_cache"]["query_vectors"][cached_hash] = {
            "hash": cached_hash,
            "model": "BAAI/bge-small-en-v1.5",
            "query_text": "cached query",
            "prefixed_text": "query: cached query",
            "embedding": [0.1, 0.2, 0.3],
        }
        queries = [
            {"query_text": "cached query"},
            {"query_text": "missing query"},
        ]

        provider_calls = {"count": 0}

        class DummyModel:
            pass

        def fake_provider():
            provider_calls["count"] += 1
            return DummyModel()

        original_encode = self.mod.encode_queries

        def fake_encode(_model, texts, batch_size=8, max_length=None):
            self.assertEqual(texts, ["missing query"])
            return np.asarray([[0.4, 0.5, 0.6]], dtype=np.float32)

        self.mod.encode_queries = fake_encode
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = pathlib.Path(tmp) / "config.yaml"
                stats = self.mod.hydrate_query_embeddings_from_config(
                    config=cfg,
                    queries=queries,
                    model_name="BAAI/bge-small-en-v1.5",
                    model_provider=fake_provider,
                    batch_size=8,
                    max_length=None,
                    config_path=str(path),
                )
                self.assertEqual(stats["hits"], 1)
                self.assertEqual(stats["misses"], 1)
                self.assertEqual(stats["written"], 1)
                self.assertEqual(provider_calls["count"], 1)
                self.assertTrue(isinstance(queries[0]["query_embedding"], np.ndarray))
                self.assertTrue(isinstance(queries[1]["query_embedding"], np.ndarray))
                self.assertEqual(
                    len(cfg["subscriptions"]["embedding_cache"]["query_vectors"]),
                    2,
                )
        finally:
            self.mod.encode_queries = original_encode

    def test_save_config_with_embedding_cache_keeps_embedding_on_one_line(self):
        cfg = {
            "subscriptions": {
                "embedding_cache": {
                    "version": 1,
                    "query_vectors": {
                        "abc": {
                            "embedding": [0.1, 0.2, 0.3],
                        }
                    },
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "config.yaml"
            ok = self.mod.save_config_with_embedding_cache(cfg, str(path))
            self.assertTrue(ok)
            text = path.read_text(encoding="utf-8")
            self.assertIn("embedding: [0.1, 0.2, 0.3]", text)
            loaded = yaml.safe_load(text)
            self.assertEqual(
                loaded["subscriptions"]["embedding_cache"]["query_vectors"]["abc"]["embedding"],
                [0.1, 0.2, 0.3],
            )


if __name__ == "__main__":
    unittest.main()
