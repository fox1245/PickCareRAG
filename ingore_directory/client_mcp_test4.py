# Create server parameters for stdio connection
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_teddynote import logging
from asyncio.exceptions import CancelledError
import os
from fastmcp import Client

# Load environment variables
if load_dotenv():
    print(".env loaded")
else:
    print("Warning: .env loading failed. Check if .env file exists and is valid.")

# Set up logging
logging.langsmith("MCP Project Client")

# --- Helper functions for JSON saving/loading and token tracking (unmodified) ---
def track_token_usage(loaded_response, cumulative_tokens=0):
    try:
        total_this_time = 0
        messages = loaded_response.get('messages', [])
        for msg in messages:
            if 'response_metadata' in msg and isinstance(msg['response_metadata'], dict) and 'token_usage' in msg['response_metadata'] and isinstance(msg['response_metadata']['token_usage'], dict):
                total_this_time += msg['response_metadata']['token_usage'].get('total_tokens', 0)
            elif 'usage_metadata' in msg and isinstance(msg['usage_metadata'], dict):
                total_this_time += msg['usage_metadata'].get('total_tokens', 0)
        
        cumulative_tokens += total_this_time
        print(f"Tokens used this time: {total_this_time}")
        print(f"Cumulative tokens: {cumulative_tokens} (estimated cost: ${cumulative_tokens * 0.00001:.4f} assuming $0.01/1M tokens)")
        return total_this_time, cumulative_tokens
    except Exception as e:
        print(f"Error tracking tokens: {str(e)}")
        return 0, cumulative_tokens

def load_agent_response_from_json(filename='agent_response.json'):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            loaded_response = json.load(f)
        if not isinstance(loaded_response, dict) or 'messages' not in loaded_response or not isinstance(loaded_response['messages'], list):
            raise ValueError("Invalid JSON structure: Must have 'messages' as list.")
        print(f"Successfully loaded agent_response from {filename}.")
        return loaded_response
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filename}.")
        return None
    except Exception as e:
        print(f"Error loading JSON: {str(e)}")
        return None

def save_agent_response_to_json(agent_response, filename='agent_response.json', indent=4):
    try:
        if not isinstance(agent_response, dict) or 'messages' not in agent_response:
            raise ValueError("Invalid agent_response: Must be a dict with 'messages' key.")
        
        serialized_messages = []
        for msg in agent_response['messages']:
            if isinstance(msg, BaseMessage):
                msg_dict = {
                    'type': msg.__class__.__name__,
                    'content': msg.content if hasattr(msg, 'content') else None,
                    'tool_calls': msg.tool_calls if hasattr(msg, 'tool_calls') else None,
                    'additional_kwargs': msg.additional_kwargs if hasattr(msg, 'additional_kwargs') else {},
                    'response_metadata': msg.response_metadata if hasattr(msg, 'response_metadata') else {},
                    'id': msg.id if hasattr(msg, 'id') else None,
                    'usage_metadata': msg.usage_metadata if hasattr(msg, 'usage_metadata') else None,
                }
                serialized_messages.append(msg_dict)
            else:
                serialized_messages.append(str(msg))
        
        serialized_response = {
            'messages': serialized_messages,
            **{k: v for k, v in agent_response.items() if k != 'messages'}
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serialized_response, f, ensure_ascii=False, indent=indent)
        
        print(f"Successfully saved agent_response to {filename} as JSON.")
        return True
    
    except Exception as e:
        print(f"Error saving to JSON: {str(e)}")
        return False

# --- Server configuration (the problematic part) ---
server_params = StdioServerParameters(
    command="python",
    args=["./mcp_test.py"],
)

# --- The refactored do() function ---
async def do():
    """
    Run the agent with an HWP file request.
    This uses stdio_client to communicate with the mcp_test.py subprocess.
    """
    print("Starting 'do()' function...")
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                tools = await load_mcp_tools(session)
                
                agent = create_react_agent("openai:gpt-5", tools)
                
                hwp_path = r"Q:\Coding\PickCareRAG\디지털 정부혁신 추진계획.hwp"
                QA = "요약하세요"
                
                print(f"Invoking agent with HWP file: {hwp_path} and QA: {QA}")
                agent_response = await agent.ainvoke({"messages": f"{hwp_path} 의 파일을 {QA}"})
                
                print("Agent response received:")
                print(agent_response)

    except CancelledError:
        print("Asyncio task was cancelled.")
    except Exception as e:
        print(f"An error occurred in do(): {e}")

async def do2():
    """
    Run the agent with a simple math request.
    This also uses stdio_client to test the connection.
    """
    print("Starting 'do2()' function...")
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                tools = await load_mcp_tools(session)
                
                agent = create_react_agent("openai:gpt-5", tools)
                
                print("Invoking agent with simple math problem...")
                agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
                
                print("Agent response received:")
                print(agent_response)

    except CancelledError:
        print("Asyncio task was cancelled.")
    except Exception as e:
        print(f"An error occurred in do2(): {e}")

async def direct_test():
    """
    This function directly imports and calls the HWPask2 function.
    This is for direct testing and does NOT use the stdio_client communication.
    """
    print("Starting 'direct_test()' function...")
    from mcp_test import HWPask2  # Import HWPask2 function here
    
    hwp_path = r"Q:\Coding\PickCareRAG\디지털 정부혁신 추진계획.hwp"
    QA = "요약하세요"
    
    print("Calling HWPask2 directly...")
    response = await HWPask2(hwp_path, QA)
    print("Direct async response:", response)

# --- Main execution block ---
if __name__ == "__main__":
    # Test each function individually to see the behavior
    # Comment out the ones you don't want to run
    
    print("\n--- Running direct_test() ---")
    asyncio.run(direct_test())

    print("\n--- Running do2() ---")
    asyncio.run(do2())

    print("\n--- Running do() (Expected to hang or error if HWP processing fails) ---")
    asyncio.run(do())