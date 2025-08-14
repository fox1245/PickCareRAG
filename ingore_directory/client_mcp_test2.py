import asyncio
import logging
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_teddynote import logging as teddy_logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Set up logging
teddy_logging.langsmith("MCP Project Client")

load_dotenv()

async def run_agent():
    try:
        # 서버 파라미터 설정
        server_params = StdioServerParameters(
            command="python",
            args=[r"Q:\Coding\PickCareRAG\mcp_test3.py"]
        )
        logger.info("서버 파라미터 설정 완료")

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                logger.info("세션 초기화 시작")
                await session.initialize()
                logger.info("세션 초기화 완료")

                # 도구 로드
                tools = await load_mcp_tools(session)
                logger.info(f"로드된 도구: {[tool.name for tool in tools]}")

                # 에이전트 생성
                agent = create_react_agent("openai:gpt-4o-mini", tools)  # gpt-4.1 대신 gpt-4o-mini
                logger.info("에이전트 생성 완료")

                # 입력 형식 수정
                file_path = "Q:/Coding/PickCareRAG/data/전기기기요약.pdf"
                input_message = {
                    "messages": [
                        HumanMessage(
                            content=f"PDFask 도구를 호출하여 '{file_path}' 파일에 대해 다음 질문 처리: "
                                    "'문서의 성격을 서술하시오.'"
                        )
                    ]
                }
                logger.info(f"에이전트 요청: {input_message}")
                agent_response = await agent.ainvoke(input_message)
                logger.info(f"에이전트 응답: {agent_response}")
                print("에이전트 응답:", agent_response.get("messages", []))

    except Exception as e:
        logger.error(f"에러 발생: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(run_agent())