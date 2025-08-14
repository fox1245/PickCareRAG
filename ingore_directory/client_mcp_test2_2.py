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
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG로 변경하여 상세 로그
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("client_mcp_test2.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# LangSmith 로깅
teddy_logging.langsmith("MCP Project Client")

load_dotenv()

async def run_agent():
    try:
        server_params = StdioServerParameters(
            command="python",
            args=[r"Q:\Coding\PickCareRAG\test_mcp4.py"]
        )
        logger.info("서버 파라미터 설정 완료")

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                logger.info("세션 초기화 시작")
                await session.initialize()
                logger.info("세션 초기화 완료")

                tools = await load_mcp_tools(session)
                logger.info(f"로드된 도구: {[tool.name for tool in tools]}")

                agent = create_react_agent(
                    model="openai:gpt-4o-mini",
                    tools=tools,
                    config={"configurable": {"retry_policy": {"max_attempts": 1}}}
                )
                logger.info("에이전트 생성 완료")

                file_path = r"Q:/Coding/PickCareRAG/data/SPRI_AI_Brief_2023년12월호_F.pdf"
                input_message = {
                    "messages": [
                        HumanMessage(
                            content=f"PDFask 도구를 호출하여 '{file_path}' 파일에 대해 다음 질문 처리: "
                                    "'문서의 성격을 서술하시오.'"
                        )
                    ]
                }
                logger.info(f"에이전트 요청: {input_message}")
                async with asyncio.timeout(180):
                    agent_response = await agent.ainvoke(input_message)
                logger.info(f"에이전트 응답: {agent_response}")
                print("에이전트 응답:", agent_response.get("messages", []))

    except asyncio.TimeoutError:
        logger.error("요청 타임아웃 발생")
        raise
    except Exception as e:
        logger.error(f"에러 발생: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(run_agent())
    
    
    
    
    # ... (기존 임포트 등)

async def run_agent():
    try:
        # ... (서버 파라미터 등 기존)

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # ... (초기화 등 기존)

                agent = create_react_agent(
                    model="openai:gpt-4o-mini",
                    tools=tools,
                    config={"configurable": {"retry_policy": {"max_attempts": 3, "wait_exponential_multiplier": 1000}}}  # 재시도 3회 추가
                )
                logger.info("에이전트 생성 완료")

                # ... (input_message 기존)

                async with asyncio.timeout(300):  # 타임아웃 180 → 300으로 증가 (테스트용)
                    agent_response = await agent.ainvoke(input_message)
                    logger.debug(f"중간 응답: {agent_response.get('intermediate_steps', [])}")  # 중간 단계 로그 추가
                logger.info(f"에이전트 응답: {agent_response}")
                print("에이전트 응답:", agent_response.get("messages", []))

    except asyncio.TimeoutError:
        logger.error("요청 타임아웃 발생 – 서버 로그 확인 요망")
        raise
    except Exception as e:
        logger.error(f"에러 발생: {e}", exc_info=True)
        raise