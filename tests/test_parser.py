from simplexid.parser import Token, ParsedSentence, parse


def test_parse_returns_parsed_sentence():
    sentence = "ac tidak dingin"

    result = parse(sentence)

    assert isinstance(result, ParsedSentence)


def test_parsed_sentence_contains_tokens():
    sentence = "ac tidak dingin"

    result = parse(sentence)

    assert isinstance(result.tokens, list)
    assert len(result.tokens) > 0


def test_token_structure():
    sentence = "ac tidak dingin"

    result = parse(sentence)

    for token in result.tokens:
        assert isinstance(token, Token)

        assert isinstance(token.id, int)
        assert isinstance(token.text, str)
        assert isinstance(token.lemma, str)
        assert isinstance(token.pos, str)
        assert isinstance(token.dep, str)
        assert isinstance(token.head_idx, int)


def test_token_ids_are_unique():
    sentence = "ac tidak dingin"

    result = parse(sentence)

    token_ids = [token.id for token in result.tokens]

    assert len(token_ids) == len(set(token_ids))


def test_parser_handles_multiple_indonesian_sentences():
    sentences = [
        "ac tidak dingin",
        "kamarnya bersih dan nyaman",
        "pelayanannya sangat baik",
        "wifi tidak stabil",
    ]

    for sentence in sentences:
        result = parse(sentence)

        assert isinstance(result, ParsedSentence)
        assert len(result.tokens) > 0