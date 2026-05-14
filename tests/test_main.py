from investing_agents.__main__ import create_agent_card


def test_create_agent_card_uses_provided_url():
    url = "http://example.com/"
    card = create_agent_card(url=url)
    assert card.url == url


def test_create_agent_card_has_expected_skills():
    card = create_agent_card()
    skill_ids = {skill.id for skill in card.skills}
    assert len(card.skills) == 5
    assert skill_ids == {
        "portfolio_management",
        "risk_analysis",
        "market_analysis",
        "financial_planning",
        "stock_analysis",
    }
