"""Dense-only retrieval for Chinese queries with a multilingual embedder. Run: python -m pytest tests"""
import pytest

import config
from src import retrieve as R


@pytest.mark.parametrize("mode,model,query,expected", [
    ("auto", "BAAI/bge-m3", "研究招募了多少人？", True),
    ("auto", "BAAI/bge-small-en-v1.5", "研究招募了多少人？", False),
    ("auto", "BAAI/bge-m3", "How many participants were recruited?", False),
    ("on", "BAAI/bge-small-en-v1.5", "研究招募了多少人？", True),
    ("off", "BAAI/bge-m3", "研究招募了多少人？", False),
])
def test_dense_only_for(monkeypatch, mode, model, query, expected):
    monkeypatch.setattr(config, "CJK_QUERY_DENSE_ONLY", mode)
    monkeypatch.setattr(config, "EMBED_MODEL", model)
    assert R.dense_only_for(query) is expected


def test_bge_m3_takes_no_query_instruction(monkeypatch):
    from src import embedder as E
    seen = []
    emb = E.Embedder.__new__(E.Embedder)
    emb.model_name = "BAAI/bge-m3"
    emb._encode = lambda texts, prefix="": seen.append(prefix) or []
    emb.encode_queries(["q"])
    emb.model_name = "BAAI/bge-small-en-v1.5"
    emb.encode_queries(["q"])
    assert seen == ["", config.QUERY_INSTRUCTION]
