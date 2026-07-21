"""llm_json: shared balanced-scan JSON extraction for LLM output."""
from defectlens.llm_json import (
    balanced_array_candidates,
    balanced_json_candidates,
    parse_string_array,
)


def test_balanced_json_candidates_ignores_braces_in_strings():
    raw = 'prose {"a": "has } brace"} tail {"b": 2}'
    assert balanced_json_candidates(raw) == ['{"a": "has } brace"}', '{"b": 2}']


def test_balanced_array_candidates_ignores_brackets_in_strings():
    raw = 'x ["a]b", "c"] y'
    assert balanced_array_candidates(raw) == ['["a]b", "c"]']


def test_parse_string_array_bare():
    assert parse_string_array('["crack near window", "damp smell"]') == [
        "crack near window",
        "damp smell",
    ]


def test_parse_string_array_fenced_and_prose():
    raw = 'Here you go:\n```json\n["a", "b"]\n```'
    assert parse_string_array(raw) == ["a", "b"]


def test_parse_string_array_drops_non_strings_and_blanks():
    assert parse_string_array('["a", 3, "", "b"]') == ["a", "b"]


def test_parse_string_array_unparseable_returns_none():
    assert parse_string_array("no json here") is None
