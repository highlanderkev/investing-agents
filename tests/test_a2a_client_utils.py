import asyncio

import pytest

import investing_agents.a2a_client_utils as a2a_client_utils
from investing_agents.a2a_client_utils import AgentTarget, extract_text_values, parse_prompt_lines


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


@pytest.mark.asyncio
async def test_run_prompt_applies_wall_clock_timeout(monkeypatch):
    class FakeResolver:
        def __init__(self, *, httpx_client, base_url):
            self.httpx_client = httpx_client
            self.base_url = base_url

        async def get_agent_card(self):
            return object()

    class SlowClient:
        def __init__(self, *, httpx_client, agent_card):
            self.httpx_client = httpx_client
            self.agent_card = agent_card

        async def send_message_streaming(self, _request):
            await asyncio.sleep(0.05)
            yield {"text": "late reply"}

    monkeypatch.setattr(a2a_client_utils, "A2ACardResolver", FakeResolver)
    monkeypatch.setattr(a2a_client_utils, "A2AClient", SlowClient)

    result = await a2a_client_utils.run_prompt(
        prompt="Hello",
        target=AgentTarget(name="Slow Target", url="http://localhost:9999"),
        timeout_s=0.01,
    )

    assert not result.success
    assert result.error == "Timed out waiting for agent response"
    assert result.response_text == ""
