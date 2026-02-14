"""Main entry point for the Investment Strategy A2A Server.

This module sets up and runs the A2A server for investment strategy agents.
"""

import os
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from investing_agents.agent_executor import InvestmentAgentExecutor


def create_agent_card(url: str = 'http://localhost:8000/') -> AgentCard:
    """Create the agent card describing the investment agent's capabilities.
    
    Args:
        url: The URL where the agent is accessible.
        
    Returns:
        AgentCard describing the investment agent.
    """
    # Define the skills this agent can perform
    portfolio_skill = AgentSkill(
        id='portfolio_management',
        name='Portfolio Management and Diversification',
        description='Provides guidance on portfolio diversification, asset allocation, and investment strategy',
        tags=['portfolio', 'diversification', 'asset allocation', 'investment strategy'],
        examples=[
            'How should I diversify my investment portfolio?',
            'What is a good asset allocation strategy?',
            'Help me build a balanced portfolio'
        ],
    )
    
    risk_skill = AgentSkill(
        id='risk_analysis',
        name='Investment Risk Analysis',
        description='Analyzes investment risk and provides risk management strategies',
        tags=['risk', 'risk management', 'volatility', 'conservative investing'],
        examples=[
            'How do I assess investment risk?',
            'What are conservative investment options?',
            'How can I reduce portfolio risk?'
        ],
    )
    
    market_skill = AgentSkill(
        id='market_analysis',
        name='Market and Stock Analysis',
        description='Provides insights on stock market investing, equity analysis, and market trends',
        tags=['stocks', 'equity', 'market analysis', 'trading'],
        examples=[
            'How do I evaluate stocks?',
            'What should I know about stock market investing?',
            'Explain value investing vs growth investing'
        ],
    )
    
    planning_skill = AgentSkill(
        id='financial_planning',
        name='Investment Planning and Strategy',
        description='Helps with long-term investment planning and financial goal setting',
        tags=['planning', 'strategy', 'long-term investing', 'financial goals'],
        examples=[
            'How should I plan for retirement?',
            'What is a good long-term investment strategy?',
            'Help me set investment goals'
        ],
    )
    
    # Create the agent card
    agent_card = AgentCard(
        name='Investment Strategy Agent',
        description='An AI-powered agent that provides investment advice, portfolio management guidance, '
                   'risk analysis, and market insights. Helps users make informed investment decisions '
                   'through comprehensive financial analysis.',
        url=url,
        version='0.1.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[portfolio_skill, risk_skill, market_skill, planning_skill],
    )
    
    return agent_card


def main():
    """Run the Investment Strategy A2A Server."""
    # Get configuration from environment variables
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8000'))
    server_url = os.getenv('SERVER_URL', f'http://localhost:{port}/')
    google_api_key = os.getenv('GOOGLE_API_KEY')
    
    # Create the agent card
    agent_card = create_agent_card(url=server_url)
    
    # Create the agent executor
    agent_executor = InvestmentAgentExecutor(api_key=google_api_key)
    
    # Create the request handler with task store
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore(),
    )
    
    # Create the A2A server application
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    print(f"Starting Investment Strategy A2A Server on {host}:{port}")
    print(f"Server URL: {server_url}")
    if google_api_key:
        print("✓ Google API Key configured - Gemini AI enabled")
    else:
        print("⚠ No Google API Key found - Using basic responses")
        print("  Set GOOGLE_API_KEY environment variable to enable AI-powered analysis")
    
    # Run the server
    uvicorn.run(server.build(), host=host, port=port)


if __name__ == '__main__':
    main()
