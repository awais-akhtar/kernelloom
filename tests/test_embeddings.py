from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from kernelloom.embeddings import FastEmbedEmbedder


class FakeTextEmbedding:
    def __init__(self, *, model_name, **options) -> None:
        self.model_name = model_name
        self.options = options

    def embed(self, texts, *, batch_size):
        return ([float(len(text)), float(batch_size)] for text in texts)


class FastEmbedTests(unittest.TestCase):
    def test_optional_fastembed_backend_is_cpu_rag_compatible(self) -> None:
        module = SimpleNamespace(TextEmbedding=FakeTextEmbedding)
        with patch.dict(sys.modules, {"fastembed": module}):
            embedder = FastEmbedEmbedder("test-local-onnx", threads=2, batch_size=8)
            self.assertEqual(embedder.embed("hello"), [5.0, 8.0])
            self.assertEqual(embedder.embed_many(["a", "abcd"]), [[1.0, 8.0], [4.0, 8.0]])
            self.assertEqual(embedder.info()["dimensions"], 2)
            self.assertEqual(embedder.warmup()["backend"], "fastembed")

    def test_missing_fastembed_has_actionable_install_message(self) -> None:
        with patch.dict(sys.modules, {"fastembed": None}):
            with self.assertRaisesRegex(RuntimeError, r"kernelloom\[fastembed\]"):
                FastEmbedEmbedder()


if __name__ == "__main__":
    unittest.main()
