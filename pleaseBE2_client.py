import asyncio
import logging
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_teddynote import logging as teddy_logging
import os  # 추가: 환경 변수 설정 용

# langsmith 완전 비활성화 (추가: 최상단)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""  # 엔드포인트 비우기 (추가 보안)

# 로깅 설정 수정: 콘솔 핸들러 제거 (파일만), level=INFO 유지
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("client_mcp_test2.log", encoding='utf-8')]  # 수정: StreamHandler 제거, utf-8 인코딩
)
logger = logging.getLogger(__name__)

# teddy_logging.langsmith("MCP Project Client") 호출 제거 (또는 주석 처리)
# teddy_logging.langsmith("MCP Project Client")  # 제거: langsmith 로깅 비활성화

load_dotenv()

# 콘솔 출력 인코딩 utf-8로 설정 (추가: 이모지 에러 방지)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

async def run_agent(QA, file_path):
    try:
        client = MultiServerMCPClient(
            {
                "pdf_rag": {"url": "http://localhost:8000/mcp/", "transport": "streamable_http"},
            }
        )
        # logger.info 제거: 콘솔 깨끗하게 (주석 처리)
        # logger.info("MultiServerMCPClient 초기화 완료")

        tools = await client.get_tools()
        if not tools:
            raise ValueError("도구 로드 실패: 서버 연결 확인 필요")
        # logger.info(f"로드된 도구: {[tool.name for tool in tools]}")  # 주석 처리

        agent = create_react_agent(
            model="openai:gpt-4o-mini",
            tools=tools,
            config={"configurable": {"retry_policy": {"max_attempts": 3, "wait_exponential_multiplier": 1000}}}
        )
        # logger.info("에이전트 생성 완료")

        content = "PDFask 도구를 호출하여 '{}' 파일에 대해 다음 질문 처리: '{}'".format(file_path, QA)
        input_message = {"messages": [HumanMessage(content=content)]}
        # logger.info(f"에이전트 요청: {input_message}")  # 주석 처리

        async with asyncio.timeout(300):
            agent_response = await agent.ainvoke(input_message)
            # logger.debug(f"중간 응답: {agent_response.get('intermediate_steps', [])}")  # 주석 처리

        # logger.info(f"에이전트 응답: {agent_response}")  # 주석 처리 (이게 UnicodeError 유발자)

        final_message = agent_response.get("messages", [])[-1]
        if hasattr(final_message, 'content'):
            print("\nAI 응답:", final_message.content, "\n")
        else:
            print("\nAI 응답: (내용 없음)\n")

    except asyncio.TimeoutError:
        print("타임아웃 에러: 서버 확인하세요.")
    except Exception as e:
        print(f"에러: {e} – 로그 파일 확인하세요.")

if __name__ == "__main__":
    while True:
        try:
            print("질문 내용을 입력해주세요: ")
            QA = sys.stdin.readline().strip()
            print()
            file_path = input("질문할 파일의 경로 입력:")
            if not QA:
                print("빈 질문입니다. 다시 입력해주세요.")
                continue
            asyncio.run(run_agent(QA, file_path))
        except KeyboardInterrupt:
            print("\n프로그램 종료. 안녕히 가세요!")
            break
        except Exception as runtime_exception:
            print(f"런타임 에러: {runtime_exception}")