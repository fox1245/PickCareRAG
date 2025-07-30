import init
init.load_dotenv()


chat = init.ChatXAI(
    xai_api_key = init.os.getenv('GROK_API_KEY'),
    model = "grok-4",
    
)


for m in chat.stream("Tell me fun thing to do in NYC"):
    print(m.content, end = "", flush = True)