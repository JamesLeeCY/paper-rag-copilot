"""Judge prompt v3e and translate-first judging (no model, no corpus)."""
import json

import src.verify as V


def test_v3e_is_v3_plus_the_english_instruction():
    v3, v3e = V.VERIFY_PROMPTS["v3"], V.VERIFY_PROMPTS["v3e"]
    assert v3e.replace(V._V3E_RULE, "").replace(
        '"reason": "brief reason in English"', '"reason": "簡短理由"') == v3
    assert "翻成英文" in v3e and "reason 一律用英文寫" in v3e


class FakeLLM:
    def __init__(self, out="Anxiety decreased.", fail=False):
        self.out, self.fail, self.calls = out, fail, 0

    def complete(self, system, user, max_tokens=300):
        self.calls += 1
        if self.fail:
            raise OSError("timeout")
        return self.out


def test_translate_only_chinese_claims_and_cache(monkeypatch):
    monkeypatch.setattr(V, "_translations", {})
    llm = FakeLLM()
    assert V.translate_claim("Anxiety decreased.", llm) == "Anxiety decreased." and llm.calls == 0
    assert V.translate_claim("焦慮下降了。", llm) == "Anxiety decreased."
    V.translate_claim("焦慮下降了。", llm)
    assert llm.calls == 1


def test_translation_failure_keeps_the_original(monkeypatch):
    monkeypatch.setattr(V, "_translations", {})
    assert V.translate_claim("焦慮下降了。", FakeLLM(fail=True)) == "焦慮下降了。"


def test_judge_eval_reads_an_item_file(tmp_path, monkeypatch):
    import eval.judge_eval as J

    f = tmp_path / "items.jsonl"
    f.write_text(json.dumps({"id": "x1", "claim": "c", "passage": "p", "gold_label": "supported",
                             "perturbation": "real", "split": "test"}) + "\n", encoding="utf-8")
    seen = {}
    monkeypatch.setattr(J, "collect_votes", lambda specs, items, prompts, tag: seen.update(
        items=items, tag=tag) or {})
    monkeypatch.setattr(J.config, "REPORT_DIR", tmp_path)
    J.main(["--items", str(f), "--split", "test", "--judges", "ollama:x"])
    assert [it["id"] for it in seen["items"]] == ["x1"] and seen["tag"] == ""
