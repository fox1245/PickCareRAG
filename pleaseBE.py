import asyncio
import logging
from mcp.server.fastmcp import FastMCP
import pdfLoader2 as PDF  # 기존 모듈 유지
import pdfLoader2 as PDF  # 기존 모듈 유지
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pathlib import Path
from dotenv import load_dotenv
from functools import lru_cache

# 로깅 설정 (상세 로그 강화)
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG로 변경하여 더 세밀한 추적
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_mcp4.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()
logger.info("환경 변수 로드 완료")

mcp = FastMCP("Simple RAG")
retriever_cache = {}  # 기존 캐시 유지

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@lru_cache(maxsize=10)
def create_vectorstore(file_path: str):  # 수정: split_docs 인자 제거, 키를 file_path만으로 단순화
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    # split_docs는 PDFask 내부에서 생성하므로, 여기서는 임시로 빈 리스트 사용 – 실제 생성은 외부에서
    # 하지만 캐싱 목적상, 함수 시그니처를 file_path만으로 해서 해시 문제 피함
    # (실제 벡터스토어는 PDFask에서 생성 후 캐시)
    logger.debug(f"Creating vectorstore for {file_path} (cached if hit)")
    # 여기에 split_docs 로직을 넣지 않음 – 외부에서 주입

@mcp.tool()
async def PDFask(file_path: str, QA: str, prompt: str = None) -> str:
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
            loader = PDF.pdfLoader(file_path=file_path, extract_bool=False)
            split_docs = await loader.load_and_split(
                text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            )
            logger.info(f"Split into {len(split_docs)} chunks")

            bm25_retriever = BM25Retriever.from_documents(split_docs)
            bm25_retriever.k = 3

            # 수정: create_vectorstore 호출 시 split_docs 직접 넘김 (캐싱 키는 file_path만)
            # lru_cache가 file_path로 캐싱되지만, 실제 생성은 split_docs 사용
            create_vectorstore(file_path)  # 캐싱 체크 (하지만 실제 생성은 아래)
            faiss_vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings(model="text-embedding-3-large"))  # 직접 생성, lru_cache는 상태 확인용으로
            logger.debug("FAISS vectorstore created successfully")  # 성공 로그 추가
            faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})

            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
            retriever_cache[cache_key] = ensemble_retriever
            logger.info("Created and cached new retriever")

        if prompt is None:
            prompt = await asyncio.to_thread(hub.pull, "rlm/rag-prompt")

        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            timeout=30,  # API 호출 타임아웃 30초
            max_retries=2  # 재시도 2회
        )
        rag_chain = (
            {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        async with asyncio.timeout(45):  # RAG 체인 타임아웃 45초 (기존 유지)
            logger.debug(f"Invoking rag_chain with QA: {QA}")
            response = await rag_chain.ainvoke(QA)
            logger.debug(f"rag_chain response: {response}")

        response_buff = [
            f"PDF Path: {file_path}",
            f"[HUMAN]\n{QA}\n",
            f"[AI]\n{response}"
        ]
        result = "\n".join(response_buff)
        logger.info(f"PDFask 결과: {result}")
        return result

    except asyncio.TimeoutError:
        logger.error(f"Timeout in PDFask for {file_path}")
        raise ValueError("PDF processing or LLM call timed out")
    except TypeError as te:  # 추가: unhashable 에러 fallback
        logger.error(f"Hash error in vectorstore: {te} – Falling back to non-cached creation")
        # fallback: 캐싱 없이 직접 생성 (필요 시)
        faiss_vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings(model="text-embedding-3-large"))
    except Exception as e:
        logger.error(f"Error in PDFask: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # HTTP 서버로 변경 (포트 기본 8000, 필요 시 --port 옵션 추가 가능)
    mcp.run(transport="streamable-http")
    logger.info("HTTP MCP 서버 시작: http://localhost:8000/mcp/")