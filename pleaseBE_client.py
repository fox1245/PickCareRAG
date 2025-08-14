import asyncio
import logging
import sys  # 추가: stdin.readline() 위해
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_teddynote import logging as teddy_logging

# 로깅 설정 수정: 콘솔 핸들러 제거 (로그 파일로만), 레벨 INFO로 낮춤 – 입력/출력 깨끗하게
logging.basicConfig(
    level=logging.INFO,  # DEBUG → INFO: 불필요 로그 줄임
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("client_mcp_test2.log")]  # StreamHandler 제거: 콘솔 오염 방지
)
logger = logging.getLogger(__name__)

# LangSmith 로깅 (필요 시 유지, 하지만 콘솔 영향 최소화)
teddy_logging.langsmith("MCP Project Client")

load_dotenv()

async def run_agent(QA):
    try:
        client = MultiServerMCPClient(
            {
                "pdf_rag": {"url": "http://localhost:8000/mcp/", "transport": "streamable_http"},
            }
        )
        logger.info("MultiServerMCPClient 초기화 완료")

        tools = await client.get_tools()
        if not tools:
            raise ValueError("도구 로드 실패: 서버 연결 확인 필요")
        logger.info(f"로드된 도구: {[tool.name for tool in tools]}")

        agent = create_react_agent(
            model="openai:gpt-4o-mini",
            tools=tools,
            config={"configurable": {"retry_policy": {"max_attempts": 3, "wait_exponential_multiplier": 1000}}}
        )
        logger.info("에이전트 생성 완료")

        file_path = r"Q:/Coding/PickCareRAG/data/SPRI_AI_Brief_2023년12월호_F.pdf"
        content = "PDFask 도구를 호출하여 '{}' 파일에 대해 다음 질문 처리: '{}'".format(file_path, QA)
        input_message = {"messages": [HumanMessage(content=content)]}
        logger.info(f"에이전트 요청: {input_message}")

        async with asyncio.timeout(300):
            agent_response = await agent.ainvoke(input_message)
            logger.debug(f"중간 응답: {agent_response.get('intermediate_steps', [])}")

        logger.info(f"에이전트 응답: {agent_response}")
        
        # 수정: 최종 AI 응답만 추출해 깔끔 출력 (messages[-1].content)
        final_message = agent_response.get("messages", [])[-1]
        if hasattr(final_message, 'content'):
            print("\nAI 응답:", final_message.content, "\n")  # 센스 있게 포맷팅
        else:
            print("\nAI 응답: (내용 없음)\n")

    except asyncio.TimeoutError:
        logger.error("요청 타임아웃 발생 – 서버 로그와 네트워크 확인 요망")
        print("타임아웃 에러: 서버 확인하세요.")
    except Exception as e:
        logger.error(f"에러 발생: {e}", exc_info=True)
        print(f"에러: {e} – 로그 파일 확인하세요.")

if __name__ == "__main__":
    while True:
        try:
            # 수정: input() 대신 sys.stdin.readline() – asyncio 콘솔 지연 방지
            QA = sys.stdin.readline().strip()
            if not QA:
                print("빈 질문입니다. 다시 입력해주세요.")
                continue
            asyncio.run(run_agent(QA))
        except KeyboardInterrupt:
            print("\n프로그램 종료. 안녕히 가세요!")
            break
        except Exception as runtime_exception:
            print(f"런타임 에러: {runtime_exception}")
            logger.error(f"런타임 에러: {runtime_exception}", exc_info=True)