# Create server parameters for stdio connection
import asyncio  # 비동기 실행을 위해
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage  # 추가: 메시지 타입 확인 위해 (핵심 추출 시 유용)
from langchain_teddynote import logging
from dotenv import load_dotenv
import traceback  # 추가: traceback 로그용
from langchain_core.exceptions import LangChainException
from asyncio.exceptions import CancelledError
import os
import mcp_test
from fastmcp import Client
load_dotenv()

logging.langsmith("MCP Project Client")


def track_token_usage(loaded_response, cumulative_tokens=0):
    """
    JSON 로드된 agent_response에서 토큰 사용량 추적 및 누적하는 함수.
    - loaded_response: load_agent_response_from_json() 반환 dict.
    - cumulative_tokens: 이전 누적 토큰 (기본 0, 세션 유지 시 사용).
    반환: (total_tokens 이번 회, cumulative_tokens 업데이트).
    """
    try:
        total_this_time = 0
        messages = loaded_response.get('messages', [])
        for msg in messages:
            # response_metadata 체크 강화: dict 여부 확인 (끈질기게 안전하게)
            if 'response_metadata' in msg and isinstance(msg['response_metadata'], dict) and 'token_usage' in msg['response_metadata'] and isinstance(msg['response_metadata']['token_usage'], dict):
                total_this_time += msg['response_metadata']['token_usage'].get('total_tokens', 0)
            # usage_metadata 체크 강화: dict 여부 확인
            elif 'usage_metadata' in msg and isinstance(msg['usage_metadata'], dict):
                total_this_time += msg['usage_metadata'].get('total_tokens', 0)
            # 옵션: 무시되는 msg 로그 (디버깅용, 나중 제거 가능)
            # else:
            #     print(f"Skipping token track for msg type: {msg.get('type', 'unknown')} - no valid metadata.")

        cumulative_tokens += total_this_time
        print(f"Tokens used this time: {total_this_time}")
        print(f"Cumulative tokens: {cumulative_tokens} (estimated cost: ${cumulative_tokens * 0.00001:.4f} assuming $0.01/1M tokens)")  # 비용 추정 (GPT-5 기준 조정)
        return total_this_time, cumulative_tokens
    except Exception as e:
        print(f"Error tracking tokens: {str(e)} (possibly in msg: {msg.get('type', 'unknown')})")  # 에러 핸들링 강화: msg 타입 로그
        return 0, cumulative_tokens


def load_agent_response_from_json(filename='agent_response.json'):
    """
    JSON 파일에서 agent_response를 로드하고, dict로 반환하는 함수.
    - filename: 로드할 JSON 파일명 (기본: 'agent_response.json').
    반환: 로드된 dict (성공 시), None (실패 시).
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            loaded_response = json.load(f)
        
        # 검증: 'messages' 키와 리스트 확인 (끈질기게 안전하게)
        if not isinstance(loaded_response, dict) or 'messages' not in loaded_response or not isinstance(loaded_response['messages'], list):
            raise ValueError("Invalid JSON structure: Must have 'messages' as list.")
        
        print(f"Successfully loaded agent_response from {filename}.")  # 로그: 성공 확인
        return loaded_response
    
    except FileNotFoundError:
        print(f"Error: File {filename} not found. Check if it was saved correctly.")  # 에러 핸들링
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filename}.")  # JSON 파싱 에러
        return None
    except Exception as e:
        print(f"Error loading JSON: {str(e)}")
        return None


def save_agent_response_to_json(agent_response, filename='agent_response.json', indent=4):
    """
    Agent response를 JSON 직렬화 가능한 형태로 변환하고, 파일로 저장하는 함수.
    - agent_response: LangGraph의 ainvoke() 반환값 (dict with 'messages').
    - filename: 저장할 JSON 파일명 (기본: 'agent_response.json').
    - indent: JSON 포맷팅 indent (기본: 4, 읽기 쉽게).
    반환: 성공 시 True, 실패 시 False.
    """
    try:
        if not isinstance(agent_response, dict) or 'messages' not in agent_response:
            raise ValueError("Invalid agent_response: Must be a dict with 'messages' key.")
        
        # 메시지 리스트를 직렬화 가능한 dict 리스트로 변환 (끈질기게 세밀하게)
        serialized_messages = []
        for msg in agent_response['messages']:
            if isinstance(msg, BaseMessage):
                # BaseMessage의 주요 속성 추출 (content, type, additional_kwargs 등)
                msg_dict = {
                    'type': msg.__class__.__name__,  # e.g., 'AIMessage', 'HumanMessage'
                    'content': msg.content if hasattr(msg, 'content') else None,
                    'tool_calls': msg.tool_calls if hasattr(msg, 'tool_calls') else None,
                    'additional_kwargs': msg.additional_kwargs if hasattr(msg, 'additional_kwargs') else {},
                    'response_metadata': msg.response_metadata if hasattr(msg, 'response_metadata') else {},
                    'id': msg.id if hasattr(msg, 'id') else None,
                    'usage_metadata': msg.usage_metadata if hasattr(msg, 'usage_metadata') else None,
                    # 필요 시 더 추가 (e.g., tool_call_id for ToolMessage)
                }
                serialized_messages.append(msg_dict)
            else:
                # 예상치 못한 타입: 그대로 dict로 변환 시도
                serialized_messages.append(str(msg))  # 안전 fallback
        
        # 전체 response 변환
        serialized_response = {
            'messages': serialized_messages,
            # 다른 키 (e.g., 만약 추가 키 있으면) 복사
            **{k: v for k, v in agent_response.items() if k != 'messages'}
        }
        
        # JSON 파일로 저장
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serialized_response, f, ensure_ascii=False, indent=indent)
        
        print(f"Successfully saved agent_response to {filename} as JSON.")  # 로그: 성공 확인
        return True
    
    except Exception as e:
        print(f"Error saving to JSON: {str(e)}")  # 에러 핸들링: 디버깅 용이
        return False



if load_dotenv():
    print(".env loaded")
else:
    print("Warning: .env loading failed. Check if .env file exists and is valid.")

server_params = StdioServerParameters(
    command="python",
    args=["./mcp_test.py"],  # 절대 경로 추천: e.g., "Q:/Coding/PickCareRAG/mcp_test.py"
)




async def do():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            #Initialize the connection
            await session.initialize()
            
            
            #Get tools
            tools = await load_mcp_tools(session)
            
            #Creat and run the test
            agent = create_react_agent("openai:gpt-5", tools)
            hwp_path = r"Q:\Coding\PickCareRAG\디지털 정부혁신 추진계획.hwp"
            QA = "요약하세요"
            agent_response = await agent.ainvoke({"messages": f"{hwp_path} 의 파일을 {QA}"})
            print(agent_response)
            
async def do2():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            #Initialize the connection
            await session.initialize()
            
            
            #Get tools
            tools = await load_mcp_tools(session)
            
            #Creat and run the test
            agent = create_react_agent("openai:gpt-5", tools)
            agent_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            print(agent_response)
              
    
        
    


        
        
        
        
async def direct_test():
    hwp_path = r"Q:\Coding\PickCareRAG\디지털 정부혁신 추진계획.hwp"
    QA = "요약하세요"
    response = await mcp_test.HWPask2(hwp_path, QA)
    print("Direct async response:", response)




if __name__ == "__main__":
    hwp_path = r"Q:\Coding\PickCareRAG\디지털 정부혁신 추진계획.hwp"
    QA = "요약하세요"
    asyncio.run(direct_test())
    asyncio.run(do2())
    #asyncio.run(do())