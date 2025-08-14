import asyncio
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient  # 새로 임포트
# from langchain_mcp_adapters.tools import load_mcp_tools  # 이건 더 이상 필요 없음, 제거
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
        # MultiServerMCPClient로 HTTP 서버 연결 (graph.py 예시 기반)
        client = MultiServerMCPClient(
            {
                "pdf_rag": {  # 서버 이름 임의 지정
                    "url": "http://localhost:8000/mcp/",  # 서버의 MCP 엔드포인트
                    "transport": "streamable_http",
                }
            }
        )
        logger.info("MultiServerMCPClient 초기화 완료")

        # 도구 로드 수정: load_mcp_tools 대신 client.get_tools() 사용 (에러 해결 핵심)
        tools = await client.get_tools()
        if not tools:
            raise ValueError("도구 로드 실패: 서버 연결 확인 필요")
        logger.info(f"로드된 도구: {[tool.name for tool in tools]}")  # 성공 로그 강화

        # 에이전트 생성 (재시도 정책 강화)
        agent = create_react_agent(
            model="openai:gpt-4o-mini",
            tools=tools,
            config={"configurable": {"retry_policy": {"max_attempts": 3, "wait_exponential_multiplier": 1000}}}
        )
        logger.info("에이전트 생성 완료")

        file_path = r"Q:/Coding/PickCareRAG/data/SPRI_AI_Brief_2023년12월호_F.pdf"
        input_message = {
            "messages": [
                HumanMessage(
                    content=f"PDFask 도구를 호출하여 '{file_path}' 파일에 대해 다음 질문 처리: "
                            "'삼성전자가 개발한 AI의 이름은?'"
                )
            ]
        }
        logger.info(f"에이전트 요청: {input_message}")

        async with asyncio.timeout(300):  # 타임아웃 증가 (테스트용)
            agent_response = await agent.ainvoke(input_message)
            logger.debug(f"중간 응답: {agent_response.get('intermediate_steps', [])}")  # 중간 단계 로그
        logger.info(f"에이전트 응답: {agent_response}")
        print("에이전트 응답:", agent_response.get("messages", []))

    except asyncio.TimeoutError:
        logger.error("요청 타임아웃 발생 – 서버 로그와 네트워크 확인 요망")
        raise
    except Exception as e:
        logger.error(f"에러 발생: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(run_agent())