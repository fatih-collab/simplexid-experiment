import pandas as pd

from simplexid.config import VARIANTS
from simplexid.dam import choose_candidate
from simplexid.dam_builder import BuiltDam, HOASA_CATEGORIES, build_dam, load_dam, normalize_text, usable_reviews, write_dam
from simplexid.parser import ParsedSentence, Token
from simplexid.pipeline import run_pipeline


def review(sentence, split="train", **labels):
    return {"sentence": sentence, "source_split": split,
            **{c: labels.get(c, "neut") for c in HOASA_CATEGORIES}}


def test_usable_reviews_excludes_text_identical_to_terma_and_masked_test():
    hoasa = pd.DataFrame([
        review("Kamar BERSIH!"), review("wifi lemot", split="validation"), review("tv rusak", split="test"),
    ])
    terma = pd.DataFrame([
        {"sentence": "kamar bersih !", "source_split": "validation"},
        {"sentence": "tv rusak", "source_split": "test"},
    ])
    assert usable_reviews(hoasa, terma)["sentence"].tolist() == ["wifi lemot"]


def test_normalize_text():
    assert normalize_text("Kamar, BERSIH!!") == "kamar bersih"


def test_build_dam_uses_lift_and_orders_by_count():
    reviews = pd.DataFrame(
        [review("lemot sekali", wifi="neg", general="pos")] * 3
        + [review("lemot", wifi="neg")] * 2
        + [review("hotel oke", general="pos")] * 5
    )
    dam = build_dam(reviews, {"lemot"}, {"hotel"}, min_count=3, min_lift=1.2)
    # P(wifi)=0.5, P(wifi|lemot)=1 -> lift 2; P(general)=0.8, P(general|lemot)=0.6 -> lift < 1
    assert dam.opinion_categories == {"lemot": ["wifi"]}
    # P(general|hotel)=1, P(general)=0.8 -> lift 1.25 < 1.5: kata umum tidak memberi konteks
    assert dam.aspect_association == {}


def test_aspect_association_keeps_strong_lift_only():
    reviews = pd.DataFrame([review("sarapan enak", sunrise_meal="pos")] * 3 + [review("kamar", general="pos")] * 7)
    dam = build_dam(reviews, set(), {"sarapan", "kamar"}, min_count=3)
    assert dam.aspect_association["sarapan"] == {"sunrise_meal": round(1.0 / 0.3, 4)}


def test_choose_candidate_uses_context_and_falls_back_to_first():
    sim = BuiltDam(aspect_association={"sarapan": {"sunrise_meal": 0.9}}).similarity
    assert choose_candidate(["general", "sunrise_meal"], ["sarapan"], sim) == "sunrise_meal"
    assert choose_candidate(["general", "sunrise_meal"], [], sim) == "general"
    assert choose_candidate(["general", "sunrise_meal"], ["kamar"], sim) == "general"


def _orphan_sentence():
    return ParsedSentence(tokens=[Token(idx=0, text="mantap", lemma="mantap", pos="ADJ", dep="root", head_idx=0)])


def test_r8_uses_built_dam_and_default_category():
    variant = VARIANTS["A3"]
    dam = BuiltDam(opinion_categories={"mantap": ["service"]})
    res = run_pipeline(None, variant, "hotel", {"mantap": 0.8}, parsed_sentence=_orphan_sentence(), dam=dam)
    assert res[0].pair.aspect_text == "service" and res[0].pair.rule == "R8"

    res = run_pipeline(None, variant, "hotel", {"mantap": 0.8}, parsed_sentence=_orphan_sentence(),
                       dam=BuiltDam(default_category="general"))
    assert res[0].pair.aspect_text == "general"

    assert run_pipeline(None, variant, "hotel", {}, parsed_sentence=_orphan_sentence(),
                        dam=BuiltDam(default_category="general")) == []


def test_dam_roundtrip(tmp_path):
    dam = BuiltDam({"bau": ["bau"]}, {"kamar": {"general": 0.5}}, "general")
    path = tmp_path / "dam.json"
    write_dam(dam, path)
    assert load_dam(path) == dam
