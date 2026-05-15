"""Investment Agent Executor for A2A Server.

This module implements the agent executor that handles investment-related queries
and provides financial analysis using AI.
"""

import logging
import os
import textwrap
import uuid

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

try:
    from a2a.utils import new_agent_text_message as _new_agent_text_message
except ImportError:
    _new_agent_text_message = None

try:
    from a2a.types import Message, Part, Role

    _USES_PYDANTIC_TYPES = True
    _PROTO_TEXT_PART_TYPE = None
    _PROTO_TEXT_FIELD_IS_MESSAGE = False
except ImportError:
    from a2a.types.a2a_pb2 import Message, Part, Role

    try:
        from a2a.types.a2a_pb2 import TextPart as _ProtoTextPart
    except ImportError:
        _ProtoTextPart = None

    _USES_PYDANTIC_TYPES = False
    _PROTO_TEXT_PART_TYPE = _ProtoTextPart
    _part_descriptor = getattr(Part, "DESCRIPTOR", None)
    _part_text_field = getattr(_part_descriptor, "fields_by_name", {}).get("text")
    _PROTO_TEXT_FIELD_IS_MESSAGE = bool(
        _part_text_field and _part_text_field.message_type is not None
    )

logger = logging.getLogger(__name__)

# Prompt template for AI-powered analysis
INVESTMENT_ADVISOR_SYSTEM_PROMPT = textwrap.dedent("""
    You are an investment advisor agent. Provide professional,
    informative responses about investment strategies, financial markets, and portfolio management.
    Provide a clear, helpful response focused on investment strategy and financial analysis.
    Include relevant considerations like risk management, diversification, and market trends where appropriate.
""").strip()

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", INVESTMENT_ADVISOR_SYSTEM_PROMPT),
        ("human", "{query}"),
    ]
)

# Default models per provider
_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "google": "gemini-2.0-flash",
    "azure": "gpt-4o-mini",
    "ollama": "llama3.2",
}


def _build_agent_text_message(text: str):
    """Create an agent text message across supported a2a-sdk versions."""
    if _new_agent_text_message is not None:
        return _new_agent_text_message(text)

    if _USES_PYDANTIC_TYPES:
        return Message(
            message_id=str(uuid.uuid4()),
            role=Role.agent,
            parts=[Part(root={"kind": "text", "text": text})],
        )

    return Message(
        message_id=str(uuid.uuid4()),
        role=Role.Value("ROLE_AGENT"),
        parts=[Part(text=_build_protobuf_text_part(text))],
    )


def _build_protobuf_text_part(text: str):
    """Build protobuf text part payload for supported protobuf schemas.

    Returns:
        Text payload for `Part.text` as either a protobuf `TextPart`, a dict payload,
        or a scalar string, depending on the installed `a2a-sdk` protobuf schema.
    """
    if _PROTO_TEXT_FIELD_IS_MESSAGE:
        if _PROTO_TEXT_PART_TYPE is not None:
            return _PROTO_TEXT_PART_TYPE(text=text)
        return {"text": text}
    return text


def _build_llm() -> BaseChatModel | None:
    """Build a LangChain chat model based on environment configuration.

    Reads:
        LLM_PROVIDER  — one of: openai (default), anthropic, google, azure, ollama
        LLM_MODEL     — optional model name override
        OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
        AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT,
        OLLAMA_BASE_URL (default http://localhost:11434)

    Returns:
        A configured BaseChatModel, or None if required config is missing or
        the provider package is not installed.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model = os.getenv("LLM_MODEL") or _DEFAULT_MODELS.get(provider)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning(
                "LLM_PROVIDER is set to 'openai' but OPENAI_API_KEY environment variable is not set"
            )
            return None
        try:
            from langchain_openai import ChatOpenAI  # noqa: PLC0415

            return ChatOpenAI(model=model, api_key=api_key)
        except ImportError:
            logger.warning(
                "LLM_PROVIDER is set to 'openai' but langchain-openai package is not installed. "
                "Install it with: pip install langchain-openai"
            )
            return None

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning(
                "LLM_PROVIDER is set to 'anthropic' but ANTHROPIC_API_KEY environment variable is not set"
            )
            return None
        try:
            from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

            return ChatAnthropic(model=model, api_key=api_key)
        except ImportError:
            logger.warning(
                "LLM_PROVIDER is set to 'anthropic' but langchain-anthropic package is not installed. "
                "Install it with: pip install langchain-anthropic"
            )
            return None

    if provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning(
                "LLM_PROVIDER is set to 'google' but GOOGLE_API_KEY environment variable is not set"
            )
            return None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # noqa: PLC0415

            return ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
        except ImportError:
            logger.warning(
                "LLM_PROVIDER is set to 'google' but langchain-google-genai package is not installed. "
                "Install it with: pip install langchain-google-genai"
            )
            return None

    if provider == "azure":
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not api_key or not endpoint:
            missing = []
            if not api_key:
                missing.append("AZURE_OPENAI_API_KEY")
            if not endpoint:
                missing.append("AZURE_OPENAI_ENDPOINT")
            logger.warning(
                "LLM_PROVIDER is set to 'azure' but required environment variable(s) not set: %s",
                ", ".join(missing),
            )
            return None
        try:
            from langchain_openai import AzureChatOpenAI  # noqa: PLC0415

            return AzureChatOpenAI(
                azure_deployment=model,
                api_key=api_key,
                azure_endpoint=endpoint,
            )
        except ImportError:
            logger.warning(
                "LLM_PROVIDER is set to 'azure' but langchain-openai package is not installed. "
                "Install it with: pip install langchain-openai"
            )
            return None

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            from langchain_ollama import ChatOllama  # noqa: PLC0415

            return ChatOllama(model=model, base_url=base_url)
        except ImportError:
            logger.warning(
                "LLM_PROVIDER is set to 'ollama' but langchain-ollama package is not installed. "
                "Install it with: pip install langchain-ollama"
            )
            return None

    logger.warning(
        "LLM_PROVIDER is set to '%s' which is not a recognized provider. "
        "Supported providers: openai, anthropic, google, azure, ollama",
        provider,
    )
    return None


class InvestmentAgent:
    """Investment strategy agent that provides financial analysis and advice."""

    def __init__(self):
        """Initialize the investment agent.

        LLM provider and credentials are read from environment variables:
            LLM_PROVIDER — openai (default), anthropic, google, azure, ollama
            LLM_MODEL    — optional model name override
        """
        self.llm = _build_llm()

    async def analyze(self, query: str) -> str:
        """Analyze investment-related queries.

        Args:
            query: The investment question or request from the user.

        Returns:
            Investment analysis or advice.
        """
        if self.llm:
            chain = _PROMPT | self.llm
            try:
                response = await chain.ainvoke({"query": query})
                return response.content
            except Exception as e:
                return f"Error generating AI response: {e}"
        else:
            return self._get_basic_response(query)

    def _get_basic_response(self, query: str) -> str:
        """Provide basic investment advice without AI.

        Args:
            query: The investment question.

        Returns:
            Basic investment guidance.
        """
        query_lower = query.lower()

        if any(word in query_lower for word in ["diversif", "portfolio", "allocat"]):
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

        elif any(word in query_lower for word in ["risk", "safe", "conserv"]):
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

        elif any(word in query_lower for word in ["stock", "equity", "share"]):
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

Note: For AI-powered personalized analysis, set the LLM_PROVIDER environment variable (openai, anthropic, google, azure, ollama) along with the corresponding API key to enable LLM integration.

Disclaimer: This information is for educational purposes only and should not be considered as financial advice. Always consult with a qualified financial advisor before making investment decisions."""


class InvestmentAgentExecutor(AgentExecutor):
    """Agent executor for investment strategy agent."""

    def __init__(self):
        """Initialize the executor.

        LLM provider and credentials are read from environment variables.
        See InvestmentAgent for supported providers.
        """
        self.agent = InvestmentAgent()

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
        user_query = context.get_user_input().strip() or "Hello, what can you help me with?"

        # Get the investment analysis
        result = await self.agent.analyze(user_query)

        # Send the result back through the event queue
        response = _build_agent_text_message(result)
        await event_queue.enqueue_event(response)

    async def cancel(
        self,
        context: RequestContext,  # noqa: ARG002
        event_queue: EventQueue,  # noqa: ARG002
    ) -> None:
        """Cancel the current execution.

        Args:
            context: The request context.
            event_queue: Queue for sending events.
        """
        raise Exception("Cancel operation is not supported for this agent")
