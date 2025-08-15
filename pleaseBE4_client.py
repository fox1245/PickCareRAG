import asyncio
import logging
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage  # 추가: 메시지 타입
from dotenv import load_dotenv
import os

# langsmith 비활성화 등 기존 유지
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("client_mcp_test2.log", encoding='utf-8')]
)
logger = logging.getLogger(__name__)

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

global messages
messages = []  # 추가: 대화 히스토리 유지 (list of messages)

async def run_agent(QA):
    global messages
    try:
        client = MultiServerMCPClient(
            {"pdf_rag": {"url": "http://localhost:8000/mcp/", "transport": "streamable_http"}}
        )
        tools = await client.get_tools()
        if not tools:
            raise ValueError("도구 로드 실패: 서버 연결 확인 필요")

        agent = create_react_agent(
            model="openai:gpt-4o-mini",  # 모델 수정 (기존 gpt-5-mini → gpt-4o-mini)
            tools=tools,
            config={"configurable": {"retry_policy": {"max_attempts": 3, "wait_exponential_multiplier": 1000}}}
        )

        # 추가: 이전 히스토리에 새 QA 추가
        content = "SharedFolderSearchAndRAG 도구를 호출하여 공유 폴더를 자율 검색해 적합 파일로 다음 질문 처리: '{}'".format(QA)
        messages.append(HumanMessage(content=content))

        input_message = {"messages": messages}  # 수정: 전체 히스토리 전달

        async with asyncio.timeout(300):
            agent_response = await agent.ainvoke(input_message)

        final_message = agent_response.get("messages", [])[-1]
        if hasattr(final_message, 'content'):
            response = final_message.content
            print("\nAI 응답:", response, "\n")
            # 추가: 응답을 히스토리에 추가
            messages.append(AIMessage(content=response))
        else:
            print("\nAI 응답: (내용 없음)\n")
            messages.append(AIMessage(content="(내용 없음)"))  # fallback

        # 옵션: 히스토리 길이 제한 (메모리 관리, e.g., 최근 20개)
        if len(messages) > 20:
            messages = messages[-20:]

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
            if not QA:
                print("빈 질문입니다. 다시 입력해주세요.")
                continue
            if QA.lower() == "reset":  # 추가: 히스토리 초기화 명령
                messages.clear()
                print("대화 히스토리 초기화 완료!")
                continue
            asyncio.run(run_agent(QA))
        except KeyboardInterrupt:
            print("\n프로그램 종료. 안녕히 가세요!")
            break
        except Exception as runtime_exception:
            print(f"런타임 에러: {runtime_exception}")