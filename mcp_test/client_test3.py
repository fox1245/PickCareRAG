import asyncio
from fastmcp import Client

async def example():
    async with Client("http://127.0.0.1:4200/my-custom-path") as client:
        await client.ping()
        
        
if __name__ == "__main__":
    asyncio.run(example())