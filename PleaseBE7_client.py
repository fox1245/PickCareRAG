import asyncio
import logging
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # 추가: SystemMessage
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
messages = []  # 대화 히스토리 유지

async def run_agent(QA):
    global messages
    try:
        client = MultiServerMCPClient(
            {"pdf_rag": {"url": "http://localhost:8000/mcp/", "transport": "streamable_http"}}
        )
        tools = await client.get_tools()
        if not tools:
            raise ValueError("도구 로드 실패: 서버 연결 확인 필요")

        # 추가: 시스템 메시지로 에이전트 지시 (메모리 강조)
        system_message = SystemMessage(content="너는 이전 대화 히스토리를 항상 참고해서 답변해야 해. 필요 시 도구를 호출하고, 그렇지 않으면 직접 답변해.")

        # 에이전트 생성 (시스템 메시지 추가)
        agent = create_react_agent(
            model="openai:gpt-4o-mini",
            tools=tools,
            config={"configurable": {"retry_policy": {"max_attempts": 3, "wait_exponential_multiplier": 1000}}},
            messages_modifier=system_message  # LangGraph에서 시스템 메시지 추가
        )

        # 수정: content를 단순 QA로 변경 (강제 도구 호출 제거)
        # 추가: 이전 히스토리를 요약해 QA에 붙임 (도구 호출 시 맥락 제공)
        history_summary = "\n".join([f"{msg.type}: {msg.content[:100]}" for msg in messages[-5:]])  # 최근 5개 요약
        enhanced_QA = f"이전 히스토리 요약: {history_summary}\n현재 질문: {QA}"
        messages.append(HumanMessage(content=enhanced_QA))

        input_message = {"messages": messages}  # 전체 히스토리 전달

        logger.info(f"에이전트 입력: {input_message}")  # 로그 강화

        async with asyncio.timeout(300):
            agent_response = await agent.ainvoke(input_message)

        final_message = agent_response.get("messages", [])[-1]
        if hasattr(final_message, 'content'):
            response = final_message.content
            print("\nAI 응답:", response, "\n")
            messages.append(AIMessage(content=response))
        else:
            print("\nAI 응답: (내용 없음)\n")
            messages.append(AIMessage(content="(내용 없음)"))

        logger.info(f"에이전트 응답: {response}")  # 로그 강화

        # 히스토리 길이 제한 (최근 10개로 강화)
        if len(messages) > 10:
            messages = messages[-10:]

    except asyncio.TimeoutError:
        print("타임아웃 에러: 서버 확인하세요.")
    except Exception as e:
        print(f"에러: {e} – 로그 파일 확인하세요.")
        logger.error(f"런타임 에러: {e}", exc_info=True)

if __name__ == "__main__":
    while True:
        try:
            print("질문 내용을 입력해주세요: ")
            QA = sys.stdin.readline().strip()
            print()
            if not QA:
                print("빈 질문입니다. 다시 입력해주세요.")
                continue
            if QA.lower() == "reset":
                messages.clear()
                print("대화 히스토리 초기화 완료!")
                continue
            asyncio.run(run_agent(QA))
        except KeyboardInterrupt:
            print("\n프로그램 종료. 안녕히 가세요!")
            break
        except Exception as runtime_exception:
            print(f"런타임 에러: {runtime_exception}")