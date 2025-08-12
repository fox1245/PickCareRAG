# Create server parameters for stdio connection
import asyncio  # 추가: 비동기 실행을 위해
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

if load_dotenv():
    print(".env loaded")
else:
    print("Warning: .env loading failed. Check if .env file exists and is valid.")

server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["./mcp_test.py"],
)

async def test():
    try:
        async with stdio_client(server_params) as (read, write):
            print("Connected to server via stdio.")  # 추가: 연결 확인 로그
            async with ClientSession(read, write) as session:
                # Initialize the connection
                await session.initialize()
                print("Session initialized.")  # 추가: 초기화 확인

                # Get tools
                tools = await load_mcp_tools(session)
                print(f"Tools loaded: {len(tools)} tools available.")  # 추가: 툴 로딩 확인

                # Create and run the agent
                # gpt-5가 아직 없으므로 gpt-4o로 변경 추천; 필요 시 "openai:gpt-5"로 되돌림
                agent = create_react_agent("openai:gpt-5", tools)
                print("Agent created.")  # 추가: 에이전트 생성 확인

                agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
                print("Agent response:", agent_response)
    except Exception as e:
        print(f"Error during execution: {str(e)}")  # 추가: 에러 핸들링

if __name__ == "__main__":  # 수정: 올바른 if 조건
    asyncio.run(test())  # 수정: 비동기 실행