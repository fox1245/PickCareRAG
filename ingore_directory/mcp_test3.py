import asyncio
import logging
from mcp.server.fastmcp import FastMCP
import pdfLoader2 as PDF  # 가정: PDF 로더 모듈
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # 변경: AsyncChatOpenAI 대신 ChatOpenAI
from langchain.retrievers import EnsembleRetriever
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("Simple RAG")

# 캐싱: 리트리버 재사용
retriever_cache = {}

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# mcp_test3.py의 PDFask 도구 수정
@mcp.tool()
async def PDFask(file_path: str, QA: str, prompt: str = None):
    try:
        file_path = Path(file_path).resolve()
        if not file_path.exists() or file_path.suffix.lower() != '.pdf':
            raise ValueError(f"Invalid PDF file: {file_path}")
        logger.info(f"Processing PDF: {file_path}")

        cache_key = str(file_path)
        if cache_key in retriever_cache:
            ensemble_retriever = retriever_cache[cache_key]
            logger.info("Using cached retriever")
        else:
            loader = PDF.pdfLoader(file_path=file_path, extract_bool=True)
            split_docs = await loader.load_and_split(  # 직접 비동기 호출
                text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            )

            bm25_retriever = BM25Retriever.from_documents(split_docs)
            bm25_retriever.k = 3
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            faiss_vectorstore = await asyncio.to_thread(FAISS.from_documents, split_docs, embeddings)
            faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})

            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
            retriever_cache[cache_key] = ensemble_retriever
            logger.info("Created and cached new retriever")

        if prompt is None:
            prompt = await asyncio.to_thread(hub.pull, "rlm/rag-prompt")

        llm = ChatOpenAI(model_name="gpt-4o-mini")
        rag_chain = (
            {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        response = await rag_chain.ainvoke(QA)
        response_buff = [
            f"PDF Path: {file_path}",
            f"[HUMAN]\n{QA}\n",
            f"[AI]\n{response}"
        ]
        return "\n".join(response_buff)

    except Exception as e:
        logger.error(f"Error in PDFask: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    mcp.run(transport="stdio")