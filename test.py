import init
import WebBaseLoader as WB
# API 키 정보 로드
init.load_dotenv()

#init.logging.langsmith("Pickcare-RAG")


def WebLoad(url = "https://n.news.naver.com/article/437/0000378416"):
    parseMan = WB.WebBaseLoader(url)
    docs = parseMan.parse()

    #문서분할
    text_splitter = init.RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)

    splits = text_splitter.split_documents(docs)

    #벡터스토어 생성
    vectorstore = init.FAISS.from_documents(documents = splits, embedding = init.OpenAIEmbeddings())


WebLoad(url = "https://n.news.naver.com/article/437/0000378416")
