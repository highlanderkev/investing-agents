from investing_agents.a2a_client_utils import extract_text_values, parse_prompt_lines


def test_parse_prompt_lines_ignores_blanks():
    raw = "first prompt\n\n  second prompt  \n   \nthird"
    assert parse_prompt_lines(raw) == ["first prompt", "second prompt", "third"]


def test_extract_text_values_finds_nested_text_keys():
    payload = {
        "event": {
            "message": {
                "parts": [
                    {"text": "hello"},
                    {"data": {"text": "world"}},
                    {"other": [1, {"text": "!"}]},
                ]
            }
        }
    }

    assert extract_text_values(payload) == ["hello", "world", "!"]
