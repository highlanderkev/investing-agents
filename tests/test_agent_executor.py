import pytest

from investing_agents.agent_executor import InvestmentAgent, LocalDocumentRetriever, _build_llm


@pytest.fixture
def agent():
    return InvestmentAgent()


def test_fallback_response_portfolio_topic(agent):
    response = agent._get_basic_response("How should I diversify my portfolio allocation?")
    assert "Diversification Advice" in response


def test_fallback_response_risk_topic(agent):
    response = agent._get_basic_response("I want a conservative and safe strategy")
    assert "Risk Management in Investing" in response


def test_fallback_response_stock_topic(agent):
    response = agent._get_basic_response("How do I evaluate a stock before buying?")
    assert "Stock Market Investment Guidance" in response


def test_fallback_response_default_topic(agent):
    response = agent._get_basic_response("Hello there")
    assert "Welcome to the Investment Strategy Agent!" in response


def test_build_llm_unknown_provider_returns_none(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert _build_llm() is None


def test_build_llm_openai_without_api_key_returns_none(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _build_llm() is None


def test_local_document_retriever_returns_relevant_chunk(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "10k.txt").write_text(
        "Revenue grew strongly year over year while operating margin improved.",
        encoding="utf-8",
    )
    (docs_dir / "notes.txt").write_text(
        "This file discusses gardening and home decoration.",
        encoding="utf-8",
    )

    retriever = LocalDocumentRetriever(str(docs_dir))
    chunks = retriever.retrieve("How did revenue and margin change?")

    assert chunks
    assert "Revenue grew strongly" in chunks[0]


@pytest.mark.asyncio
async def test_fallback_response_includes_rag_context(monkeypatch, tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "statement.txt").write_text(
        "Cash flow from operations increased while debt declined.",
        encoding="utf-8",
    )

    monkeypatch.setenv("RAG_DOCUMENTS_PATH", str(docs_dir))
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    agent = InvestmentAgent()
    response = await agent.analyze("What changed in cash flow and debt?")

    assert "Retrieved financial document context" in response
    assert "Cash flow from operations increased" in response
