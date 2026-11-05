import pandas as pd
import pytest

from simplexid.config import VARIANTS
from simplexid.evaluation.sc_targeted import SHEET_COLUMNS, discordant_pairs, evaluate_annotations, read_sheet
from simplexid.parser import ParsedSentence, Token


def mock(spec):
    return ParsedSentence(tokens=[Token(idx=i, text=t, lemma=t, pos=p, dep=d, head_idx=h)
                                  for i, (t, p, d, h) in enumerate(spec)])


# "sinyal sangat lemah dan tidak stabil"
ASYM = mock([
    ("sinyal", "NOUN", "root", 0), ("sangat", "ADV", "advmod", 2), ("lemah", "ADJ", "amod", 0),
    ("dan", "CCONJ", "cc", 5), ("tidak", "PART", "advmod", 5), ("stabil", "ADJ", "conj", 2),
])
LEX = {"lemah": -0.4, "stabil": 0.4}


def _build():
    return discordant_pairs(["s1"], {"s1": ASYM}, {"s1": "sinyal sangat lemah dan tidak stabil"},
                            VARIANTS["A5"], VARIANTS["A6"], LEX, lambda sid: frozenset({"sinyal"}))


def test_only_discordant_pairs_are_selected():
    sheet, system = _build()
    assert sheet["item_id"].tolist() == ["s1__a0__o5"]   # "lemah" sama di kedua sistem, tidak terpilih
    assert system.iloc[0]["pred_without_sc"] == "POSITIVE" and system.iloc[0]["pred_with_sc"] == "NEGATIVE"


def test_sheet_does_not_reveal_predictions():
    sheet, _ = _build()
    assert list(sheet.columns) == SHEET_COLUMNS
    assert not any("pred" in c for c in sheet.columns)


def _annotated(rows):
    return pd.DataFrame([{"item_id": i, "is_valid_pair": v, "gold_polarity": p, "annotation_status": s}
                         for i, v, p, s in rows])


SYSTEM = pd.DataFrame([
    {"item_id": "a", "pred_without_sc": "POSITIVE", "pred_with_sc": "NEGATIVE"},
    {"item_id": "b", "pred_without_sc": "POSITIVE", "pred_with_sc": "NEGATIVE"},
    {"item_id": "c", "pred_without_sc": "POSITIVE", "pred_with_sc": "NEGATIVE"},
    {"item_id": "d", "pred_without_sc": "POSITIVE", "pred_with_sc": "NEGATIVE"},
    {"item_id": "e", "pred_without_sc": "POSITIVE", "pred_with_sc": "NEGATIVE"},
])


def test_evaluate_counts_hand_computed():
    sheet = _annotated([
        ("a", "YES", "NEGATIVE", "REVIEWED"),   # hanya SC benar
        ("b", "YES", "POSITIVE", "REVIEWED"),   # hanya pewarisan benar
        ("c", "YES", "NEUTRAL", "REVIEWED"),    # keduanya salah
        ("d", "NO", "", "REVIEWED"),            # bukan pasangan: dikeluarkan
        ("e", "", "", ""),                      # belum direview: diabaikan
    ])
    r = evaluate_annotations(sheet, SYSTEM)
    assert (r["reviewed"], r["evaluated"], r["invalid_pair"]) == (4, 3, 1)
    assert (r["only_sc_correct"], r["only_inheritance_correct"], r["both_wrong"]) == (1, 1, 1)
    assert r["p_value"] == 1.0


def test_evaluate_rejects_invalid_values():
    with pytest.raises(ValueError, match="is_valid_pair"):
        evaluate_annotations(_annotated([("a", "MAYBE", "", "REVIEWED")]), SYSTEM)
    with pytest.raises(ValueError, match="gold_polarity"):
        evaluate_annotations(_annotated([("a", "YES", "", "REVIEWED")]), SYSTEM)


def test_read_sheet_accepts_semicolon_from_excel(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("item_id;is_valid_pair\na;YES\n", encoding="utf-8-sig")
    assert read_sheet(path).to_dict("records") == [{"item_id": "a", "is_valid_pair": "YES"}]


def test_ctx_targeted_evaluate_hand_computed():
    import importlib.util, pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "experiments" / "run_ctx_targeted.py"
    spec = importlib.util.spec_from_file_location("run_ctx_targeted", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    sheet = pd.DataFrame([
        {"item_id": "a", "decision": "IMPLICIT", "gold_aspect_category": "general", "annotation_status": "REVIEWED"},
        {"item_id": "b", "decision": "IMPLICIT", "gold_aspect_category": "wifi", "annotation_status": "REVIEWED"},
        {"item_id": "c", "decision": "NOT_IMPLICIT", "gold_aspect_category": "", "annotation_status": "REVIEWED"},
        {"item_id": "d", "decision": "", "gold_aspect_category": "", "annotation_status": ""},
    ])
    system = pd.DataFrame([{"item_id": i, "pred_static": "general", "pred_contextual": "wifi"} for i in "abcd"])
    r = mod.evaluate(sheet, system)
    assert (r["reviewed"], r["evaluated"], r["only_static_correct"], r["only_contextual_correct"]) == (3, 2, 1, 1)
    sheet.loc[0, "gold_aspect_category"] = "harga"
    with pytest.raises(ValueError, match="ontologi"):
        mod.evaluate(sheet, system)
