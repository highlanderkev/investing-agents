import pytest

from investing_agents.agent_executor import InvestmentAgent, _build_llm


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
