import asyncio
from fastmcp import Client

async def example(name :str):
    async with Client("http://127.0.0.1:8000/mcp") as client:
        result = await client.call_tool("hello", {"name": name})
        print(result)
        
        
        
if __name__ == "__main__":
    asyncio.run(example(name ="booleanjars.com"))