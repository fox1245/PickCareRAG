from fastmcp import FastMCP


#Create a basic server instance
mcp = FastMCP(name = "MyAssitantServer")

# You can also add instructions for how to interact with the server

mcp_with_instructions = FastMCP(
    name = "HelpfulAssistant",
    instructions="""
        This server provides 
    """
)

# tool은 클라이언트가 작업을 수행하거나 외부 시스템에 엑세스하기 위해 호출 할 수 있는 기능이다.
@mcp.tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers together."""
    return a * b


# 프롬프트는 LLM에게 안내하기 위한 재사용 가능한 메시지 템플릿이다.
@mcp.prompt
def analyze_data(data_points: list[float]) -> str:
    """Creates a prompt asking for analysis of numerical data."""
    formatted_data = ", ".join(str(point) for point in data_points)
    return f"Please analyze these data points: {formatted_data}"


#FastMCP는 구성 가능한 include/exclude 태그 세트를 기반으로 구성 요소를 선택적으로 노출하는 태그 기반 필터링을 지원함. 이는 다양한 환경이나 사용자에 맞게 
#서버의 다양한 뷰를 생성하는 데 유용함.
#매개변수를 사용하여 정의할 경우 구성 요소에 태그를 지정할 수 있다.


@mcp.tool
def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"



if __name__ == "__main__":
    #mcp.run(transport="http", host = "127.0.0.1", port = 9000)
    mcp.run()
    