# Investment Strategy A2A Agent

An Agent-to-Agent (A2A) protocol server that provides investment strategy advice, portfolio management guidance, risk analysis, and market insights.

## Overview

This project implements an A2A-compliant agent for investment and financial strategy using Python and the `uv` package manager. The agent can:

- 🎯 **Portfolio Management**: Provide diversification strategies and asset allocation advice
- ⚖️ **Risk Analysis**: Assess investment risk and recommend risk management approaches  
- 📈 **Market Analysis**: Offer insights on stock market investing and equity analysis
- 📊 **Financial Planning**: Guide long-term investment planning and goal setting

The agent uses Google's Gemini AI (when configured) for intelligent, context-aware responses, and falls back to structured guidance when AI is not available.

## Architecture

This implementation follows the A2A protocol specification and is based on the [a2a-samples](https://github.com/a2aproject/a2a-samples) repository structure.

**Key Components:**
- `agent_executor.py`: Core investment analysis logic and agent executor
- `__main__.py`: A2A server setup with agent card and capabilities
- `test_client.py`: Example client for testing the agent

## Prerequisites

- Python 3.12 or higher
- [UV](https://docs.astral.sh/uv/) package manager
- (Optional) Google API Key for Gemini AI integration

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

## Configuration

### Optional: Enable AI-Powered Analysis

To enable Gemini AI for more intelligent responses:

1. Get a Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

2. Set the environment variable:
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

3. Or create a `.env` file:
   ```bash
   echo "GOOGLE_API_KEY=your-api-key-here" > .env
   ```

### Server Configuration

Configure the server using environment variables:

- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `SERVER_URL`: Public URL for the agent card (default: `http://localhost:8000/`)
- `GOOGLE_API_KEY`: Google API key for Gemini (optional)

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
✓ Google API Key configured - Gemini AI enabled
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

The agent provides four main skills:

1. **Portfolio Management and Diversification**
2. **Investment Risk Analysis**  
3. **Market and Stock Analysis**
4. **Investment Planning and Strategy**

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

## Disclaimer

**Important:** This agent provides general investment information and educational content only. It is **not** a substitute for professional financial advice. 

- The information provided should not be considered as financial or investment advice
- Always consult with a qualified financial advisor before making investment decisions
- Past performance does not guarantee future results
- All investments carry risk, including potential loss of principal

The sample code is provided for demonstration purposes to illustrate the A2A protocol mechanics. When building production applications, treat any external agent as potentially untrusted and implement appropriate security measures.

## License

See [LICENSE](LICENSE) file for details.
