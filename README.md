# Investment Strategy A2A Agent

An Agent-to-Agent (A2A) protocol server that provides investment strategy advice, portfolio management guidance, risk analysis, and market insights.

## Overview

This project implements an A2A-compliant agent for investment and financial strategy using Python and the `uv` package manager. The agent can:

- 🎯 **Portfolio Management**: Provide diversification strategies and asset allocation advice
- ⚖️ **Risk Analysis**: Assess investment risk and recommend risk management approaches  
- 📈 **Market Analysis**: Offer insights on stock market investing and equity analysis
- 📊 **Financial Planning**: Guide long-term investment planning and goal setting
- 🔍 **Stock Analysis**: Analyze specific stocks, company fundamentals, and sentiment

The agent uses [LangChain](https://python.langchain.com/) to integrate with multiple LLM providers (OpenAI, Anthropic, Google Gemini, Azure OpenAI, and Ollama) for intelligent, context-aware responses, and falls back to structured guidance when no AI provider is configured.

## Architecture

This implementation follows the A2A protocol specification and is based on the [a2a-samples](https://github.com/a2aproject/a2a-samples) repository structure.

**Key Components:**
- `agent_executor.py`: Core investment analysis logic and agent executor
- `__main__.py`: A2A server setup with agent card and capabilities
- `test_client.py`: Example client for testing the agent

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- (Optional) An API key for at least one supported LLM provider (OpenAI, Anthropic, Google, Azure OpenAI) or a running [Ollama](https://ollama.com/) instance

## Installation

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/highlanderkev/investing-agents.git
   cd investing-agents
   ```

3. **Install dependencies**:
   ```bash
   uv sync
   ```

   To enable a specific LLM provider, install its optional extra:

   | Provider | Command |
   |---|---|
   | OpenAI | `uv sync --extra openai` |
   | Anthropic | `uv sync --extra anthropic` |
   | Google (Gemini) | `uv sync --extra google` |
   | Azure OpenAI | `uv sync --extra azure` |
   | Ollama | `uv sync --extra ollama` |
   | All providers | `uv sync --extra all` |

## Configuration

### Optional: Enable AI-Powered Analysis

The agent uses LangChain to support multiple LLM providers. Set `LLM_PROVIDER` to one of the supported values and supply the corresponding credentials.

#### OpenAI

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="your-api-key-here"
```

#### Anthropic

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY="your-api-key-here"
```

#### Google (Gemini)

```bash
export LLM_PROVIDER=google
export GOOGLE_API_KEY="your-api-key-here"
```

Get a Google API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

#### Azure OpenAI

```bash
export LLM_PROVIDER=azure
export AZURE_OPENAI_API_KEY="your-api-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

#### Ollama (local)

```bash
export LLM_PROVIDER=ollama
# Optionally override the base URL (default: http://localhost:11434)
export OLLAMA_BASE_URL="http://localhost:11434"
```

#### Model Override

To use a model other than the provider default, set `LLM_MODEL`:

```bash
export LLM_MODEL="gpt-4o"
```

Default models per provider:

| Provider | Default Model |
|---|---|
| openai | `gpt-4o-mini` |
| anthropic | `claude-3-5-haiku-20241022` |
| google | `gemini-2.0-flash` |
| azure | `gpt-4o-mini` |
| ollama | `llama3.2` |

You can also store your configuration in a `.env` file instead of setting environment variables:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your-api-key-here
```

### Server Configuration

Configure the server using environment variables:

- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `SERVER_URL`: Public URL for the agent card (default: `http://localhost:8000/`)
- `LLM_PROVIDER`: LLM provider to use — `openai` (default), `anthropic`, `google`, `azure`, `ollama`
- `LLM_MODEL`: Optional model name override for the selected provider
- `OPENAI_API_KEY`: OpenAI API key (required when `LLM_PROVIDER=openai`)
- `ANTHROPIC_API_KEY`: Anthropic API key (required when `LLM_PROVIDER=anthropic`)
- `GOOGLE_API_KEY`: Google API key for Gemini (required when `LLM_PROVIDER=google`)
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key (required when `LLM_PROVIDER=azure`)
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL (required when `LLM_PROVIDER=azure`)
- `OLLAMA_BASE_URL`: Ollama server URL (used when `LLM_PROVIDER=ollama`, default: `http://localhost:11434`)

## Usage

### Running the Agent Server

Start the investment strategy agent server:

```bash
uv run python -m investing_agents
```

The server will start on `http://localhost:8000` by default.

**Expected output:**
```
Starting Investment Strategy A2A Server on 0.0.0.0:8000
Server URL: http://localhost:8000/
```

### Testing with the Example Client

In a separate terminal, test the agent:

**Single Query:**
```bash
uv run python src/investing_agents/test_client.py --query "How should I diversify my portfolio?"
```

**Interactive Mode:**
```bash
uv run python src/investing_agents/test_client.py --interactive
```

### Example Queries

Try these example questions with the agent:

- "How should I diversify my investment portfolio?"
- "What are conservative investment options?"
- "Explain value investing vs growth investing"
- "How do I assess investment risk?"
- "What should I know about stock market investing?"
- "How should I plan for retirement?"

## A2A Protocol Integration

This agent implements the A2A protocol and can be integrated with any A2A-compliant host or client.

### Agent Card

The agent exposes its capabilities through an agent card at `/agent-card`:

```bash
curl http://localhost:8000/agent-card
```

### Skills

The agent provides five main skills:

1. **Portfolio Management and Diversification**
2. **Investment Risk Analysis**  
3. **Market and Stock Analysis**
4. **Investment Planning and Strategy**
5. **Stock Analysis**

### Using with Other A2A Clients

Any A2A-compliant client can interact with this agent. For example, using the [A2A CLI](https://github.com/a2aproject/a2a-samples/tree/main/samples/python/hosts/cli):

```bash
# From the a2a-samples repository
cd samples/python/hosts/cli
uv run . --url http://localhost:8000/
```

## Development

### Project Structure

```
investing-agents/
├── src/
│   └── investing_agents/
│       ├── __init__.py           # Package initialization
│       ├── __main__.py           # Server entry point
│       ├── agent_executor.py    # Investment agent logic
│       └── test_client.py       # Test client
├── pyproject.toml               # Project dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

### Running Tests

```bash
uv run pytest
```

### Code Quality

Format code:
```bash
uv run ruff format src/
```

Lint code:
```bash
uv run ruff check src/
```

## Related Resources

- [A2A Protocol Specification](https://github.com/a2aproject/A2A)
- [A2A Python SDK](https://github.com/a2aproject/a2a-python)
- [A2A Samples](https://github.com/a2aproject/a2a-samples)
- [A2A Quickstart Notebook](https://github.com/a2aproject/a2a-samples/blob/main/notebooks/a2a_quickstart.ipynb)
- [LangChain Documentation](https://python.langchain.com/)
- [LangChain OpenAI Integration](https://python.langchain.com/docs/integrations/chat/openai/)
- [LangChain Anthropic Integration](https://python.langchain.com/docs/integrations/chat/anthropic/)
- [LangChain Google Generative AI Integration](https://python.langchain.com/docs/integrations/chat/google_generative_ai/)
- [LangChain Ollama Integration](https://python.langchain.com/docs/integrations/chat/ollama/)

## Disclaimer

**Important:** This agent provides general investment information and educational content only. It is **not** a substitute for professional financial advice. 

- The information provided should not be considered as financial or investment advice
- Always consult with a qualified financial advisor before making investment decisions
- Past performance does not guarantee future results
- All investments carry risk, including potential loss of principal

The sample code is provided for demonstration purposes to illustrate the A2A protocol mechanics. When building production applications, treat any external agent as potentially untrusted and implement appropriate security measures.

## License

See [LICENSE](LICENSE) file for details.
