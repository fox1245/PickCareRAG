import asyncio
from fastmcp import Client  # fastmcp.Client를 활용

async def example():
    async with Client("http://127.0.0.1:8000/mcp/") as client:  # 기존 Client 사용
        if await client.ping():  # ping 성공 확인
            print("Ping 성공: MCP 연결 확인됨.")
            
            # requests.get() 대신 MCP Client로 Resource 쿼리
            # 예: 서버에 'status'라는 Resource가 있다고 가정 (실제 서버 문서 확인)
            try:
                response = await client.get_resource("status")  # MCP Resource GET (fastmcp API 기반)
                print("MCP Resource 응답:", response)  # MCP 형식 데이터 출력
            except Exception as e:
                print("Resource 쿼리 실패:", str(e))
            
            # 추가: Tool 호출 예시 (POST-like)
            # tool_response = await client.call_tool("some_tool", params={})  # 필요 시

if __name__ == "__main__":
    asyncio.run(example())