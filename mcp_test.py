from mcp.server.fastmcp import FastMCP
import init
from RAG import PDFask, prompt_maker
from langchain_teddynote import logging
from dotenv import load_dotenv

load_dotenv()

logging.langsmith("MCP Project")
mcp = FastMCP("Test")

@mcp.tool()
def add(a : int, b : int ) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a : int , b : int) -> int:
    """Multiplay two numbers"""
    return a * b

@mcp.tool()
def PDFask_custom( QA : str) -> str:
    """If you read the PDF and are asking about the contents of the PDF, use this function"""
    custom_prompt = prompt_maker()
    file = "data/SPRI_AI_Brief_2023년12월호_F.pdf"
    pdf_response = PDFask(file_path = file, model = "gpt-5", QA = QA, prompt = custom_prompt, k = 3)
    res = ""
    for elem in pdf_response:
        res += str(elem)
    return res
    
    



if __name__ == "__main__":
    mcp.run(transport= "stdio")
    
