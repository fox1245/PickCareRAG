import asyncio
import logging
from mcp.server.fastmcp import FastMCP
import HWP_async as HWP
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
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# 로깅 설정 (상세 로그 강화)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_mcp4.log", encoding='utf-8'),  # 파일 utf-8
              logging.StreamHandler()],  # 콘솔 유지
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()
logger.info("환경 변수 로드 완료")

mcp = FastMCP("Simple RAG")
retriever_cache = {}  # 기존 캐시 유지

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@mcp.tool()
async def prompt_maker():  # async 유지 (MCP 도구 규칙)
    """고양이 말투로 대화할 있도록 하는 프롬프트를 반환하는 함수이다. PDFask의 prompt 인자나 HWPask의 prompt인자에 할당하게 되면 고양이 말투를 사용한다."""
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="냥! 저는 문서를 읽고 말할 줄 아는 똑똑한 고양이예요~ 😺\n{context}를 보고, {question}에 대해 최대한 귀엽고 사랑스럽고 카와이한 고양이 말투로 정리해줄게요! 야옹~ 답변은 아주 디테일하고 내 섬세한 수염처럼 초~ 센서티브하게 답변해줄게 냥냥. 답변이 만족스러우면 고급 츄르 한 개 줄래냥?. \n답변: ",
    )
    return custom_prompt  # 수정: await 제거, return으로 직접 반환

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
    
@mcp.tool()
async def HWPask(file_path: str, QA: str, prompt: str = None) -> str:
    try:
        file_path = Path(file_path).resolve()
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path} - Check path and rename if spaces present.")
            raise ValueError(f"File does not exist: {file_path}. Please verify the path and remove spaces if any.")
        if file_path.suffix.lower() not in ('.hwp', '.hwpx'):
            raise ValueError(f"Invalid HWP or HWPX file: {file_path}")
        logger.info(f"Processing HWP: {file_path}")
        logger.debug(f"File confirmed exists: {file_path.exists()}, Suffix: {file_path.suffix.lower()}")  # 강화 디버그

        cache_key = str(file_path)
        if cache_key in retriever_cache and retriever_cache[cache_key] is not None:  # 캐시 유효성 체크
            ensemble_retriever = retriever_cache[cache_key]
            logger.info("Using valid cached retriever")
        else:
            loader = HWP.HWP(file_path=file_path)
            context = await loader.load()
            if not context:  # 빈 context 방지
                raise ValueError("Loaded context is empty - Check HWPLoader or file integrity.")
            # 나머지 코드 (idx pop, split_docs 등) 기존 유지
            # ...
            
            idx = 0
            while idx < len(context):  # 안전하게 pop 위해 while 사용 (pop 시 인덱스 변함)
                if context[idx].page_content is None:
                    context.pop(idx)
                else:
                    idx += 1
                    
                    
            original_context = "".join(str(c) for c in context)  # 효율적 join
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            text_chunks = text_splitter.split_text(original_context)
            split_docs = [Document(page_content=chunk, metadata={"source": str(file_path)}) for chunk in text_chunks]
            
            logger.info(f"Split into {len(split_docs)} chunks")
            
            bm25_retriever = BM25Retriever.from_documents(split_docs)
            bm25_retriever.k = 3
            
            create_vectorstore(file_path)  # 기존 유지
            faiss_vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings(model="text-embedding-3-large"))
            logger.debug("FAISS vectorstore created successfully")
            faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})
            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
          
            
            
            retriever_cache[cache_key] = ensemble_retriever
            logger.info("Created and cached new retriever")

        # rag_chain 실행 부분 기존 유지
        # ...
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
        logger.info(f"HWPask 결과: {result}")
        return result

    except ValueError as ve:
        logger.error(f"ValueError in HWPask: {ve}", exc_info=True)
        return f"Error: {ve} - Please check file path and try again."  # 클라이언트에 명확 에러 반환
    except Exception as e:
        logger.error(f"Unexpected error in HWPask: {e}", exc_info=True)
        raise  

        
    

if __name__ == "__main__":
    # HTTP 서버로 변경 (포트 기본 8000, 필요 시 --port 옵션 추가 가능)
    mcp.run(transport="streamable-http")
    logger.info("HTTP MCP 서버 시작: http://localhost:8000/mcp/")