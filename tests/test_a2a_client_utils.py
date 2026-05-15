import asyncio

import httpx
import pytest

import investing_agents.a2a_client_utils as a2a_client_utils
from investing_agents.a2a_client_utils import (
    AgentRunResult,
    AgentTarget,
    extract_text_values,
    parse_prompt_lines,
)


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


def test_dedupe_preserve_order():
    from investing_agents.a2a_client_utils import _dedupe_preserve_order

    values = ["apple", "banana", "apple", "cherry", "banana", "date"]
    result = _dedupe_preserve_order(values)
    assert result == ["apple", "banana", "cherry", "date"]


def test_dedupe_preserve_order_empty():
    from investing_agents.a2a_client_utils import _dedupe_preserve_order

    assert _dedupe_preserve_order([]) == []


def test_normalize_error_connect_error():
    from investing_agents.a2a_client_utils import _normalize_error

    exc = httpx.ConnectError("Connection failed")
    assert _normalize_error(exc) == "Could not connect to agent server"


def test_normalize_error_timeout():
    from investing_agents.a2a_client_utils import _normalize_error

    exc = httpx.TimeoutException("Timeout")
    assert _normalize_error(exc) == "Timed out waiting for agent response"


def test_normalize_error_timeout_error():
    from investing_agents.a2a_client_utils import _normalize_error

    exc = TimeoutError("Timeout")
    assert _normalize_error(exc) == "Timed out waiting for agent response"


def test_normalize_error_generic():
    from investing_agents.a2a_client_utils import _normalize_error

    exc = ValueError("Something went wrong")
    assert _normalize_error(exc) == "ValueError: Something went wrong"


def test_summarize_results_single_target():
    from investing_agents.a2a_client_utils import summarize_results

    results = [
        AgentRunResult(
            target_name="Target A",
            target_url="http://localhost:8000",
            prompt="test",
            success=True,
            response_text="response",
            chunks=["response"],
            error=None,
            latency_ms=100.0,
            first_event_ms=50.0,
            event_count=1,
            timestamp=1234567890.0,
        ),
        AgentRunResult(
            target_name="Target A",
            target_url="http://localhost:8000",
            prompt="test2",
            success=True,
            response_text="response2",
            chunks=["response2"],
            error=None,
            latency_ms=200.0,
            first_event_ms=75.0,
            event_count=2,
            timestamp=1234567891.0,
        ),
    ]

    summary = summarize_results(results)
    assert "Target A" in summary
    assert summary["Target A"]["runs"] == 2.0
    assert summary["Target A"]["success_rate_pct"] == 100.0
    assert summary["Target A"]["avg_latency_ms"] == 150.0
    assert summary["Target A"]["avg_first_event_ms"] == 62.5


def test_summarize_results_multiple_targets():
    from investing_agents.a2a_client_utils import summarize_results

    results = [
        AgentRunResult(
            target_name="Target A",
            target_url="http://localhost:8000",
            prompt="test",
            success=True,
            response_text="response",
            chunks=["response"],
            error=None,
            latency_ms=100.0,
            first_event_ms=50.0,
            event_count=1,
            timestamp=1234567890.0,
        ),
        AgentRunResult(
            target_name="Target B",
            target_url="http://localhost:8001",
            prompt="test",
            success=False,
            response_text="",
            chunks=[],
            error="Error",
            latency_ms=300.0,
            first_event_ms=None,
            event_count=0,
            timestamp=1234567890.0,
        ),
    ]

    summary = summarize_results(results)
    assert "Target A" in summary
    assert "Target B" in summary
    assert summary["Target A"]["success_rate_pct"] == 100.0
    assert summary["Target B"]["success_rate_pct"] == 0.0
    assert summary["Target B"]["avg_first_event_ms"] == 0.0


def test_summarize_results_empty():
    from investing_agents.a2a_client_utils import summarize_results

    summary = summarize_results([])
    assert summary == {}
