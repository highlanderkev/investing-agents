"""Investment Agent Executor for A2A Server.

This module implements the agent executor that handles investment-related queries
and provides financial analysis using AI.
"""

import os
from typing import Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class InvestmentAgent:
    """Investment strategy agent that provides financial analysis and advice."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the investment agent.
        
        Args:
            api_key: Google API key for Gemini. If not provided, will use GOOGLE_API_KEY env var.
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        self.client = None
        
        if genai and self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    async def analyze(self, query: str) -> str:
        """Analyze investment-related queries.
        
        Args:
            query: The investment question or request from the user.
            
        Returns:
            Investment analysis or advice.
        """
        if self.client:
            # Use Gemini for AI-powered analysis
            prompt = f"""You are an investment advisor agent. Provide professional, 
informative responses about investment strategies, financial markets, and portfolio management.

User query: {query}

Provide a clear, helpful response focused on investment strategy and financial analysis.
Include relevant considerations like risk management, diversification, and market trends where appropriate."""
            
            try:
                response = self.client.models.generate_content(
                    model='gemini-2.0-flash-exp',
                    contents=prompt
                )
                return response.text
            except Exception as e:
                return f"Error generating AI response: {str(e)}. Please provide a GOOGLE_API_KEY environment variable."
        else:
            # Fallback to basic responses if Gemini is not available
            return self._get_basic_response(query)
    
    def _get_basic_response(self, query: str) -> str:
        """Provide basic investment advice without AI.
        
        Args:
            query: The investment question.
            
        Returns:
            Basic investment guidance.
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['diversif', 'portfolio', 'allocat']):
            return """Investment Portfolio Diversification Advice:

1. **Asset Allocation**: Consider spreading investments across different asset classes:
   - Stocks (equity) for growth potential
   - Bonds for stability and income
   - Real estate for inflation hedge
   - Cash equivalents for liquidity

2. **Geographic Diversification**: Don't limit yourself to domestic markets
   
3. **Sector Diversification**: Invest across various industries to reduce sector-specific risk

4. **Risk Assessment**: Align your portfolio with your risk tolerance and investment timeline

5. **Regular Rebalancing**: Review and adjust your portfolio periodically

Remember: Past performance doesn't guarantee future results. Consider consulting with a financial advisor for personalized advice."""

        elif any(word in query_lower for word in ['risk', 'safe', 'conserv']):
            return """Risk Management in Investing:

1. **Understand Your Risk Tolerance**: Consider your age, income, financial goals, and comfort with volatility

2. **Risk Mitigation Strategies**:
   - Diversification across assets
   - Dollar-cost averaging
   - Setting stop-loss orders
   - Regular portfolio reviews

3. **Conservative Investment Options**:
   - Government bonds
   - High-grade corporate bonds
   - Index funds
   - Money market accounts

4. **Risk vs. Return**: Higher potential returns typically come with higher risk

5. **Time Horizon**: Longer investment periods can help weather market volatility

Always assess your personal financial situation before making investment decisions."""

        elif any(word in query_lower for word in ['stock', 'equity', 'share']):
            return """Stock Market Investment Guidance:

1. **Research Before Investing**: 
   - Company fundamentals (earnings, revenue, debt)
   - Industry trends and competitive position
   - Management quality
   
2. **Investment Approaches**:
   - Value investing: Undervalued stocks with strong fundamentals
   - Growth investing: Companies with high growth potential
   - Index investing: Low-cost diversification through index funds

3. **Key Metrics to Consider**:
   - P/E ratio (Price-to-Earnings)
   - Dividend yield
   - Market capitalization
   - Revenue and earnings growth

4. **Long-term Perspective**: Avoid emotional decisions based on short-term market fluctuations

5. **Professional Guidance**: Consider consulting a financial advisor for personalized stock recommendations

Disclaimer: This is general information and not specific investment advice."""

        else:
            return """Welcome to the Investment Strategy Agent!

I can help you with:

1. **Portfolio Management**: Diversification strategies and asset allocation
2. **Risk Assessment**: Understanding and managing investment risk
3. **Investment Strategies**: Growth, value, income, and index investing
4. **Market Analysis**: General market trends and sector analysis
5. **Financial Planning**: Long-term investment planning and goal setting

What specific investment topic would you like to explore?

Note: For AI-powered personalized analysis, please configure the GOOGLE_API_KEY environment variable with your Google API key to enable Gemini integration.

Disclaimer: This information is for educational purposes only and should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions."""


class InvestmentAgentExecutor(AgentExecutor):
    """Agent executor for investment strategy agent."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the executor.
        
        Args:
            api_key: Google API key for Gemini integration.
        """
        self.agent = InvestmentAgent(api_key=api_key)

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute the investment agent with the given context.
        
        Args:
            context: The request context containing user input.
            event_queue: Queue for sending events back to the client.
        """
        # Extract the user's query from the context
        user_query = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                # Part is a RootModel, so we need to access part.root
                if hasattr(part, 'root') and hasattr(part.root, 'text') and part.root.text:
                    user_query += part.root.text + " "
        
        user_query = user_query.strip()
        if not user_query:
            user_query = "Hello, what can you help me with?"
        
        # Get the investment analysis
        result = await self.agent.analyze(user_query)
        
        # Send the result back through the event queue
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Cancel the current execution.
        
        Args:
            context: The request context.
            event_queue: Queue for sending events.
        """
        raise Exception('Cancel operation is not supported for this agent')
