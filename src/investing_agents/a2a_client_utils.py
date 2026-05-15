"""Shared A2A client utilities for interactive UI and evaluation flows."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendStreamingMessageRequest


@dataclass
class AgentTarget:
    """Addressable target for an agent server."""

    name: str
    url: str


@dataclass
class AgentRunResult:
    """Normalized result for one prompt run against one target."""

    target_name: str
    target_url: str
    prompt: str
    success: bool
    response_text: str
    chunks: list[str]
    error: str | None
    latency_ms: float
    first_event_ms: float | None
    event_count: int
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def fetch_agent_card(agent_url: str, timeout_s: float = 15.0) -> dict[str, Any]:
    """Resolve an agent card and return a compact serializable summary."""
    timeout = httpx.Timeout(timeout_s)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        resolver = A2ACardResolver(httpx_client=http_client, base_url=agent_url)
        card = await resolver.get_agent_card()

    return {
        "name": getattr(card, "name", "Unknown Agent"),
        "description": getattr(card, "description", ""),
        "version": getattr(card, "version", "unknown"),
        "url": getattr(card, "url", agent_url),
        "skills": [
            {
                "id": getattr(skill, "id", ""),
                "name": getattr(skill, "name", ""),
                "description": getattr(skill, "description", ""),
            }
            for skill in getattr(card, "skills", [])
        ],
    }


async def run_prompt(
    *,
    prompt: str,
    target: AgentTarget,
    timeout_s: float = 90.0,
) -> AgentRunResult:
    """Send a prompt to one target and normalize streamed response output."""
    started = time.perf_counter()
    first_event_ms: float | None = None
    chunks: list[str] = []
    event_count = 0

    try:
        timeout = httpx.Timeout(timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as http_client:
            resolver = A2ACardResolver(httpx_client=http_client, base_url=target.url)
            agent_card = await resolver.get_agent_card()
            client = A2AClient(httpx_client=http_client, agent_card=agent_card)

            payload = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": prompt}],
                    "messageId": uuid4().hex,
                }
            }
            request = SendStreamingMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**payload),
            )

            stream_response = client.send_message_streaming(request)
            async for event in stream_response:
                event_count += 1
                event_payload = _safe_model_dump(event)
                texts = extract_text_values(event_payload)
                if texts and first_event_ms is None:
                    first_event_ms = (time.perf_counter() - started) * 1000
                chunks.extend(texts)

        latency_ms = (time.perf_counter() - started) * 1000
        response_text = "\n".join(_dedupe_preserve_order(chunks)).strip()

        return AgentRunResult(
            target_name=target.name,
            target_url=target.url,
            prompt=prompt,
            success=True,
            response_text=response_text,
            chunks=chunks,
            error=None,
            latency_ms=latency_ms,
            first_event_ms=first_event_ms,
            event_count=event_count,
            timestamp=time.time(),
        )
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - started) * 1000
        return AgentRunResult(
            target_name=target.name,
            target_url=target.url,
            prompt=prompt,
            success=False,
            response_text="",
            chunks=chunks,
            error=_normalize_error(exc),
            latency_ms=latency_ms,
            first_event_ms=first_event_ms,
            event_count=event_count,
            timestamp=time.time(),
        )


async def run_compare(
    *,
    prompt: str,
    targets: list[AgentTarget],
    timeout_s: float = 90.0,
) -> list[AgentRunResult]:
    """Run one prompt across many targets in parallel."""
    tasks = [run_prompt(prompt=prompt, target=target, timeout_s=timeout_s) for target in targets]
    return await asyncio.gather(*tasks)


async def run_batch_evaluation(
    *,
    prompts: list[str],
    targets: list[AgentTarget],
    timeout_s: float = 90.0,
    max_concurrency: int = 4,
) -> list[AgentRunResult]:
    """Run a prompt set across targets with bounded concurrency."""
    sem = asyncio.Semaphore(max(1, max_concurrency))

    async def _run_one(prompt: str, target: AgentTarget) -> AgentRunResult:
        async with sem:
            return await run_prompt(prompt=prompt, target=target, timeout_s=timeout_s)

    tasks = [_run_one(prompt, target) for prompt in prompts for target in targets]
    return await asyncio.gather(*tasks)


def parse_prompt_lines(raw_text: str) -> list[str]:
    """Parse newline-separated prompts, trimming empty lines."""
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def run_coro_sync(coro):
    """Run an async coroutine from synchronous Streamlit code."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    if not loop.is_running():
        return loop.run_until_complete(coro)

    new_loop = asyncio.new_event_loop()
    try:
        return new_loop.run_until_complete(coro)
    finally:
        new_loop.close()


def summarize_results(results: list[AgentRunResult]) -> dict[str, dict[str, float]]:
    """Compute per-target aggregate metrics for evaluation tables."""
    bucket: dict[str, dict[str, float]] = {}
    for result in results:
        target_bucket = bucket.setdefault(
            result.target_name,
            {
                "runs": 0.0,
                "successes": 0.0,
                "latency_sum": 0.0,
                "first_event_sum": 0.0,
                "first_event_count": 0.0,
            },
        )
        target_bucket["runs"] += 1
        target_bucket["latency_sum"] += result.latency_ms
        if result.success:
            target_bucket["successes"] += 1
        if result.first_event_ms is not None:
            target_bucket["first_event_sum"] += result.first_event_ms
            target_bucket["first_event_count"] += 1

    summary: dict[str, dict[str, float]] = {}
    for target_name, stats in bucket.items():
        runs = stats["runs"]
        first_count = stats["first_event_count"]
        summary[target_name] = {
            "runs": runs,
            "success_rate_pct": (stats["successes"] / runs * 100) if runs else 0.0,
            "avg_latency_ms": (stats["latency_sum"] / runs) if runs else 0.0,
            "avg_first_event_ms": (stats["first_event_sum"] / first_count) if first_count else 0.0,
        }
    return summary


def extract_text_values(payload: Any) -> list[str]:
    """Recursively collect text snippets from A2A event payloads."""
    results: list[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "text" and isinstance(value, str) and value.strip():
                    results.append(value)
                else:
                    _walk(value)
            return

        if isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(payload)
    return results


def _safe_model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, (dict, list, str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _normalize_error(exc: Exception) -> str:
    if isinstance(exc, httpx.ConnectError):
        return "Could not connect to agent server"
    if isinstance(exc, httpx.TimeoutException):
        return "Timed out waiting for agent response"
    return f"{exc.__class__.__name__}: {exc}"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered
