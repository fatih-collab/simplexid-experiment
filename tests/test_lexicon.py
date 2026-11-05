import pytest

from simplexid.lexicon import load_lexicon, merge_lexicons, parse_binary, parse_sentistrength
from simplexid.parser import Token
from simplexid.scoring import token_score


def test_parse_sentistrength_normalizes_scale_and_handles_crlf():
    assert parse_sentistrength("bersih:5\r\nkotor:-3\r\nabu-abu:-1\r\n") == {
        "bersih": 1.0, "kotor": -0.6, "abu-abu": -0.2,
    }


def test_parse_binary_weights():
    assert parse_binary("adil\n\n", "buruk\n", weight=0.6) == {"adil": 0.6, "buruk": -0.6}


def test_merge_prefers_primary():
    assert merge_lexicons({"murah": 0.4}, {"murah": -0.6, "panas": -0.6}) == {"murah": 0.4, "panas": -0.6}


def test_third_party_file_missing_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="README"):
        load_lexicon("merged_v1", third_party_dir=tmp_path)


def test_third_party_checksum_is_verified(tmp_path):
    (tmp_path / "sentiwords_id.txt").write_text("bersih:5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum"):
        load_lexicon("ss_v1", third_party_dir=tmp_path)


def test_token_score_prefers_surface_form_then_lemma():
    token = Token(idx=0, text="Kamarnya", lemma="Kamar", pos="NOUN", dep="root", head_idx=0)
    assert token_score(token, {"kamarnya": 0.2, "kamar": 0.9}) == 0.2
    assert token_score(token, {"kamar": 0.9}) == 0.9
    assert token_score(token, {}) == 0.0
