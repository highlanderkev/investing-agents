"""Streamlit frontend for interacting with and evaluating investment agents."""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import asdict
from typing import Any

import streamlit as st

from investing_agents.a2a_client_utils import (
    AgentTarget,
    fetch_agent_card,
    parse_prompt_lines,
    run_batch_evaluation,
    run_compare,
    run_coro_sync,
    run_prompt,
    summarize_results,
)


def main() -> None:
    st.set_page_config(
        page_title="Investing Agents Workbench",
        page_icon="📈",
        layout="wide",
    )

    _init_state()

    st.title("Investing Agents Workbench")
    st.caption("Interact with A2A agents, compare outputs, and run lightweight evaluations.")

    _render_sidebar()

    chat_tab, compare_tab, eval_tab = st.tabs(["Chat", "Compare", "Evaluate"])

    with chat_tab:
        _render_chat_tab()
    with compare_tab:
        _render_compare_tab()
    with eval_tab:
        _render_evaluate_tab()


def _init_state() -> None:
    st.session_state.setdefault(
        "targets",
        [
            {
                "name": "Local Agent",
                "url": "http://localhost:8000",
                "enabled": True,
            }
        ],
    )
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("last_compare_results", [])
    st.session_state.setdefault("evaluation_rows", [])
    st.session_state.setdefault("local_processes", {})


def _render_sidebar() -> None:
    st.sidebar.header("Targets")

    with st.sidebar.expander("Target Registry", expanded=True):
        for idx, target in enumerate(st.session_state.targets):
            cols = st.columns([1.5, 2.5, 1, 0.8])
            name = cols[0].text_input(
                "Name",
                value=target["name"],
                key=f"target_name_{idx}",
                label_visibility="collapsed",
            )
            url = cols[1].text_input(
                "URL",
                value=target["url"],
                key=f"target_url_{idx}",
                label_visibility="collapsed",
            )
            enabled = cols[2].checkbox(
                "Use",
                value=target.get("enabled", True),
                key=f"target_enabled_{idx}",
                label_visibility="collapsed",
            )
            remove = cols[3].button("Remove", key=f"target_remove_{idx}")

            st.session_state.targets[idx] = {
                "name": name.strip() or f"Target {idx + 1}",
                "url": url.strip(),
                "enabled": enabled,
            }

            if remove:
                st.session_state.targets.pop(idx)
                st.rerun()

        if st.button("Add Target"):
            st.session_state.targets.append(
                {
                    "name": f"Target {len(st.session_state.targets) + 1}",
                    "url": "http://localhost:8000",
                    "enabled": True,
                }
            )
            st.rerun()

    with st.sidebar.expander("Local Helper Mode", expanded=False):
        alias = st.text_input("Alias", value="local-managed")
        provider = st.selectbox(
            "Provider",
            ["openai", "anthropic", "google", "azure", "ollama"],
            index=0,
        )
        model = st.text_input("Model Override (optional)", value="")
        host = st.text_input("Host", value="0.0.0.0")
        port = st.number_input("Port", min_value=1000, max_value=65535, value=8010, step=1)

        if st.button("Start Local Server"):
            _start_local_server(
                alias=alias.strip() or f"local-{port}",
                provider=provider,
                model=model.strip(),
                host=host.strip() or "0.0.0.0",
                port=int(port),
            )

        if st.session_state.local_processes:
            st.markdown("Running local servers")
            for server_name, proc_info in list(st.session_state.local_processes.items()):
                process = proc_info["process"]
                status = "running" if process.poll() is None else f"exited ({process.poll()})"
                st.write(f"- {server_name}: {status} at {proc_info['url']}")
                if st.button(f"Stop {server_name}", key=f"stop_{server_name}"):
                    _stop_local_server(server_name)
                    st.rerun()


def _render_chat_tab() -> None:
    st.subheader("Chat with One Agent")

    enabled_targets = _enabled_targets()
    if not enabled_targets:
        st.warning("Enable at least one target in the sidebar.")
        return

    target_names = [target.name for target in enabled_targets]
    selected_name = st.selectbox("Target", target_names, key="chat_target")
    selected_target = next(target for target in enabled_targets if target.name == selected_name)

    action_cols = st.columns([1, 1, 4])
    if action_cols[0].button("Check Agent Card"):
        card = run_coro_sync(fetch_agent_card(selected_target.url))
        st.success(f"Connected to {card['name']} (v{card['version']})")
        with st.expander("Agent Details"):
            st.write(card)

    if action_cols[1].button("Clear Chat"):
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        role = message["role"]
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(message["content"])
            metrics = message.get("metrics")
            if metrics:
                st.caption(
                    f"latency: {metrics['latency_ms']:.0f} ms | "
                    f"first event: {metrics.get('first_event_ms', 0.0):.0f} ms | "
                    f"events: {metrics['event_count']}"
                )

    prompt = st.chat_input("Ask an investment question")
    if not prompt:
        return

    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.spinner("Contacting agent..."):
        result = run_coro_sync(run_prompt(prompt=prompt, target=selected_target))

    if result.success:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result.response_text or "(No text returned)",
                "metrics": {
                    "latency_ms": result.latency_ms,
                    "first_event_ms": result.first_event_ms or 0.0,
                    "event_count": result.event_count,
                },
            }
        )
    else:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": f"Error: {result.error}",
                "metrics": {
                    "latency_ms": result.latency_ms,
                    "first_event_ms": result.first_event_ms or 0.0,
                    "event_count": result.event_count,
                },
            }
        )
    st.rerun()


def _render_compare_tab() -> None:
    st.subheader("Side-by-Side Compare")

    enabled_targets = _enabled_targets()
    if len(enabled_targets) < 1:
        st.warning("Enable at least one target in the sidebar.")
        return

    default_prompt = "How should I diversify a portfolio with moderate risk tolerance?"
    prompt = st.text_area("Prompt", value=default_prompt, height=120)

    selected_names = st.multiselect(
        "Targets",
        [target.name for target in enabled_targets],
        default=[target.name for target in enabled_targets[: min(2, len(enabled_targets))]],
    )

    timeout_s = st.slider("Timeout (seconds)", min_value=10, max_value=180, value=90, step=5)

    if st.button("Run Compare", type="primary"):
        selected_targets = [target for target in enabled_targets if target.name in selected_names]
        if not selected_targets:
            st.error("Select at least one target.")
            return

        with st.spinner("Running comparisons..."):
            results = run_coro_sync(
                run_compare(
                    prompt=prompt.strip(), targets=selected_targets, timeout_s=float(timeout_s)
                )
            )
        st.session_state.last_compare_results = [asdict(result) for result in results]

    results_payload = st.session_state.last_compare_results
    if not results_payload:
        return

    results = results_payload
    columns = st.columns(max(1, len(results)))
    for col, result in zip(columns, results, strict=False):
        with col:
            st.markdown(f"### {result['target_name']}")
            st.caption(result["target_url"])
            if result["success"]:
                st.success("Success")
                st.write(result["response_text"] or "(No text returned)")
            else:
                st.error(result["error"] or "Unknown error")

            st.caption(
                f"latency: {result['latency_ms']:.0f} ms | "
                f"first event: {(result['first_event_ms'] or 0.0):.0f} ms | "
                f"events: {result['event_count']}"
            )


def _render_evaluate_tab() -> None:
    st.subheader("Batch Evaluation")

    enabled_targets = _enabled_targets()
    if not enabled_targets:
        st.warning("Enable at least one target in the sidebar.")
        return

    seed_prompts = "\n".join(
        [
            "How should I diversify my investment portfolio?",
            "What are conservative investment options right now?",
            "Explain value investing versus growth investing.",
            "How do I assess my personal risk tolerance?",
            "What should I research before buying an individual stock?",
        ]
    )

    prompts_raw = st.text_area(
        "Prompts (one per line)",
        value=seed_prompts,
        height=200,
    )

    selected_names = st.multiselect(
        "Targets for evaluation",
        [target.name for target in enabled_targets],
        default=[target.name for target in enabled_targets[: min(2, len(enabled_targets))]],
        key="eval_targets",
    )

    controls = st.columns(3)
    timeout_s = controls[0].slider(
        "Timeout (seconds)",
        min_value=10,
        max_value=180,
        value=90,
        step=5,
        key="eval_timeout",
    )
    max_concurrency = controls[1].slider(
        "Max concurrency",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
    )

    if controls[2].button("Run Evaluation", type="primary"):
        prompts = parse_prompt_lines(prompts_raw)
        selected_targets = [target for target in enabled_targets if target.name in selected_names]

        if not prompts:
            st.error("Add at least one prompt.")
            return
        if not selected_targets:
            st.error("Select at least one target.")
            return

        with st.spinner("Running evaluation batch..."):
            results = run_coro_sync(
                run_batch_evaluation(
                    prompts=prompts,
                    targets=selected_targets,
                    timeout_s=float(timeout_s),
                    max_concurrency=max_concurrency,
                )
            )

        st.session_state.evaluation_rows = [
            {
                **asdict(result),
                "rating": 3,
                "pass_fail": "pass" if result.success else "fail",
                "review_notes": "",
            }
            for result in results
        ]

    rows: list[dict[str, Any]] = st.session_state.evaluation_rows
    if not rows:
        st.info("Run an evaluation batch to see rows and metrics.")
        return

    st.markdown("### Summary")
    summary = summarize_results([_row_to_result_like(row) for row in rows])
    metric_cols = st.columns(max(1, len(summary)))
    for col, (target_name, stats) in zip(metric_cols, summary.items(), strict=False):
        with col:
            st.metric("Target", target_name)
            st.metric("Runs", f"{stats['runs']:.0f}")
            st.metric("Success", f"{stats['success_rate_pct']:.1f}%")
            st.metric("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")

    st.markdown("### Results")
    for idx, row in enumerate(rows):
        status_emoji = "✅" if row["success"] else "❌"
        label = (
            f"{status_emoji} {row['target_name']} | prompt {idx + 1} | "
            f"latency {row['latency_ms']:.0f} ms"
        )
        with st.expander(label, expanded=False):
            st.write(f"Prompt: {row['prompt']}")
            if row["success"]:
                st.write(row["response_text"] or "(No text returned)")
            else:
                st.error(row.get("error") or "Unknown error")

            review_cols = st.columns([1, 1, 3])
            row["pass_fail"] = review_cols[0].selectbox(
                "Pass/Fail",
                options=["pass", "fail"],
                index=0 if row.get("pass_fail") == "pass" else 1,
                key=f"eval_pass_fail_{idx}",
            )
            row["rating"] = review_cols[1].slider(
                "Rating",
                min_value=1,
                max_value=5,
                value=int(row.get("rating", 3)),
                key=f"eval_rating_{idx}",
            )
            row["review_notes"] = review_cols[2].text_input(
                "Review notes",
                value=row.get("review_notes", ""),
                key=f"eval_notes_{idx}",
            )


def _enabled_targets() -> list[AgentTarget]:
    targets: list[AgentTarget] = []
    for target in st.session_state.targets:
        if target.get("enabled") and target.get("url"):
            targets.append(AgentTarget(name=target["name"], url=target["url"]))
    return targets


def _start_local_server(alias: str, provider: str, model: str, host: str, port: int) -> None:
    if alias in st.session_state.local_processes:
        st.warning(f"Server '{alias}' is already tracked.")
        return

    env = {
        **dict(**st.session_state.get("_env_copy", {})),
        **_get_os_environ(),
        "HOST": host,
        "PORT": str(port),
        "SERVER_URL": f"http://localhost:{port}/",
        "LLM_PROVIDER": provider,
    }
    if model:
        env["LLM_MODEL"] = model

    process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "investing_agents"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    url = f"http://localhost:{port}"
    st.session_state.local_processes[alias] = {
        "process": process,
        "url": url,
        "provider": provider,
        "started_at": time.time(),
    }

    st.session_state.targets.append({"name": alias, "url": url, "enabled": True})
    st.success(f"Started local server '{alias}' on {url}")


def _stop_local_server(alias: str) -> None:
    proc_info = st.session_state.local_processes.pop(alias, None)
    if not proc_info:
        return

    process = proc_info["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def _row_to_result_like(row: dict[str, Any]):
    class _ResultLike:
        def __init__(self, payload: dict[str, Any]):
            self.target_name = payload["target_name"]
            self.success = payload["success"]
            self.latency_ms = float(payload["latency_ms"])
            first_event_value = payload.get("first_event_ms")
            self.first_event_ms = (
                float(first_event_value) if first_event_value is not None else None
            )

    return _ResultLike(row)


def _get_os_environ() -> dict[str, str]:
    import os

    return dict(os.environ)


if __name__ == "__main__":
    main()
