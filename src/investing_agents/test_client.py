"""Test client for the Investment Strategy A2A Agent.

This script demonstrates how to interact with the investment agent using the A2A protocol.
"""

import asyncio
import sys
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendStreamingMessageRequest


async def test_investment_agent(query: str, agent_url: str = "http://localhost:8000") -> None:
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
            # Get the agent card first
            print("Fetching agent card...")
            resolver = A2ACardResolver(
                httpx_client=http_client,
                base_url=agent_url,
            )
            
            agent_card = await resolver.get_agent_card()
            print(f"✓ Connected to: {agent_card.name}")
            print(f"  Description: {agent_card.description}")
            print(f"  Version: {agent_card.version}")
            print(f"\n  Available Skills:")
            for skill in agent_card.skills:
                print(f"    - {skill.name}: {skill.description}")
            print()
            
            # Create the A2A client
            client = A2AClient(
                httpx_client=http_client,
                agent_card=agent_card
            )
            
            # Create a message to send to the agent
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': query}
                    ],
                    'messageId': uuid4().hex,
                },
            }
            
            streaming_request = SendStreamingMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload)
            )
            
            # Send the task and stream the response
            print("Sending query to agent...\n")
            print(f"Response:")
            print(f"{'-'*70}")
            
            stream_response = client.send_message_streaming(streaming_request)
            
            async for chunk in stream_response:
                print(chunk.model_dump(mode='json', exclude_none=True))
            
            print(f"{'-'*70}\n")
            print(f"✓ Query completed successfully")
            
    except httpx.ConnectError:
        print(f"✗ Error: Could not connect to agent at {agent_url}")
        print("  Make sure the agent server is running.")
        print("  Start it with: uv run python -m investing_agents")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


async def run_interactive_mode(agent_url: str = "http://localhost:8000") -> None:
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
        default='http://localhost:8000',
        help='URL of the investment agent server (default: http://localhost:8000)'
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
