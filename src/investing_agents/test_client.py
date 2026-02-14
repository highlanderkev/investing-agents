"""Test client for the Investment Strategy A2A Agent.

This script demonstrates how to interact with the investment agent using the A2A protocol.
"""

import asyncio
import sys
from typing import Optional

import httpx
from a2a.client.client import A2AClient
from a2a.types import ClientMessage, TextContentPart, UserMessage


async def test_investment_agent(query: str, agent_url: str = "http://localhost:8000/") -> None:
    """Test the investment agent with a query.
    
    Args:
        query: The investment question to ask.
        agent_url: URL of the investment agent server.
    """
    print(f"\n{'='*70}")
    print(f"Testing Investment Agent")
    print(f"Agent URL: {agent_url}")
    print(f"Query: {query}")
    print(f"{'='*70}\n")
    
    try:
        async with httpx.AsyncClient() as http_client:
            # Create the A2A client
            client = A2AClient(
                agent_url=agent_url,
                http_client=http_client,
            )
            
            # Get the agent card first
            print("Fetching agent card...")
            agent_card = await client.get_agent_card()
            print(f"✓ Connected to: {agent_card.name}")
            print(f"  Description: {agent_card.description}")
            print(f"  Version: {agent_card.version}")
            print(f"\n  Available Skills:")
            for skill in agent_card.skills:
                print(f"    - {skill.name}: {skill.description}")
            print()
            
            # Create a message to send to the agent
            message = ClientMessage(
                role='user',
                content=[TextContentPart(text=query)]
            )
            
            # Send the task and stream the response
            print("Sending query to agent...\n")
            print(f"Response:")
            print(f"{'-'*70}")
            
            full_response = ""
            async for event in client.task_new_iter([message]):
                if hasattr(event, 'content') and event.content:
                    for content_part in event.content:
                        if hasattr(content_part, 'text'):
                            text = content_part.text
                            print(text, end='', flush=True)
                            full_response += text
            
            print(f"\n{'-'*70}\n")
            print(f"✓ Query completed successfully")
            
    except httpx.ConnectError:
        print(f"✗ Error: Could not connect to agent at {agent_url}")
        print("  Make sure the agent server is running.")
        print("  Start it with: uv run python -m investing_agents")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


async def run_interactive_mode(agent_url: str = "http://localhost:8000/") -> None:
    """Run the client in interactive mode.
    
    Args:
        agent_url: URL of the investment agent server.
    """
    print("Investment Strategy Agent - Interactive Client")
    print("=" * 70)
    print(f"Connected to: {agent_url}")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 70)
    print()
    
    while True:
        try:
            query = input("Your question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break
            
            if not query:
                continue
            
            await test_investment_agent(query, agent_url)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            break


async def main():
    """Main entry point for the test client."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test client for Investment Strategy A2A Agent'
    )
    parser.add_argument(
        '--url',
        default='http://localhost:8000/',
        help='URL of the investment agent server (default: http://localhost:8000/)'
    )
    parser.add_argument(
        '--query',
        help='Single query to send to the agent. If not provided, runs in interactive mode.'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode (default if no --query provided)'
    )
    
    args = parser.parse_args()
    
    if args.query:
        await test_investment_agent(args.query, args.url)
    else:
        await run_interactive_mode(args.url)


if __name__ == '__main__':
    asyncio.run(main())
