import pytest

from simplexid.parser import (
    ParsedSentence,
    Token,
    UDPipeParser,
    from_conllu,
    parse,
    parse_tokens,
)


def test_parse_returns_parsed_sentence():
    result = parse("ac tidak dingin")

    assert isinstance(result, ParsedSentence)
    assert len(result.tokens) > 0


def test_token_structure():
    result = parse("ac tidak dingin")

    for token in result.tokens:
        assert isinstance(token, Token)
        assert isinstance(token.idx, int)
        assert isinstance(token.text, str)
        assert isinstance(token.lemma, str)
        assert isinstance(token.pos, str)
        assert isinstance(token.dep, str)
        assert isinstance(token.head_idx, int)


def test_token_ids_are_unique_and_sequential():
    result = parse("kamarnya bersih dan nyaman")

    assert [t.idx for t in result.tokens] == list(range(len(result.tokens)))


def test_parser_handles_multiple_indonesian_sentences():
    for sentence in [
        "ac tidak dingin",
        "kamarnya bersih dan nyaman",
        "pelayanannya sangat baik",
        "wifi tidak stabil",
    ]:
        result = parse(sentence)
        assert isinstance(result, ParsedSentence)
        assert len(result.tokens) > 0


def test_parse_tokens_produces_dependencies():
    # Regresi: versi spacy-udpipe lama tidak pernah menjalankan parser pada
    # token pretokenized, sehingga semua dep kosong dan head == diri sendiri.
    result = parse_tokens(["sinyalnya", "sangat", "lemah", "dan", "tidak", "stabil"])

    roots = [t for t in result.tokens if t.head_idx == t.idx]
    assert len(roots) == 1
    assert all(t.dep for t in result.tokens)
    assert any(t.dep == "conj" for t in result.tokens)


def test_parse_tokens_preserves_terma_tokenization():
    # Kalimat TermA bisa memuat beberapa titik; tetap harus satu pohon dan
    # jumlah/urutan token tidak boleh berubah (span index gold bergantung padanya).
    tokens = ["bersih", ".", "ramah", ".", "ac", "nya", "dingin", "."]
    result = parse_tokens(tokens)

    assert result.token_text() == tokens
    assert sum(t.head_idx == t.idx for t in result.tokens) == 1


def test_parse_tokens_rejects_tokens_with_spaces():
    with pytest.raises(ValueError):
        parse_tokens(["kamar mandi", "bersih"])


def test_conllu_roundtrip():
    result = parse_tokens(["wifi", "tidak", "stabil"])

    assert from_conllu(result.to_conllu()) == result


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        UDPipeParser(tmp_path / "missing.udpipe")


def test_corrupt_model_gives_clear_error(tmp_path):
    fake = tmp_path / "fake.udpipe"
    fake.write_text("<!DOCTYPE html><html></html>", encoding="utf-8")
    with pytest.raises(RuntimeError):
        UDPipeParser(fake)
