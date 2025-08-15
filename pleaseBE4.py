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
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableMap
from datetime import datetime
import os
import re

# 로깅 설정 (상세 로그 강화)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("test_mcp4.log", encoding='utf-8'),  # 파일 utf-8
              logging.StreamHandler()],  # 콘솔 유지
)
logger = logging.getLogger(__name__)

SHARED_FOLDER_PATH = os.getenv("SHARED_FOLDER_PATH", r"Q:\Coding\PickCareRAG\shared")  #

# 환경 변수 로드
load_dotenv()
logger.info("환경 변수 로드 완료")

mcp = FastMCP("Simple RAG")
retriever_cache = {}  # 기존 캐시 유지
vectorstore_cache = {}  # 신규: vectorstore 캐시 (hash 에러 피함)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 수정: 글로벌 RunnableLambda로 정의해서 모든 도구에서 공유
format_docs_runnable = RunnableLambda(format_docs)

@mcp.tool()
async def prompt_maker():  # async 유지 (MCP 도구 규칙)
    """고양이 말투로 대화할 있도록 하는 프롬프트를 반환하는 함수이다. PDFask의 prompt 인자나 HWPask의 prompt인자에 할당하게 되면 고양이 말투를 사용한다."""
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="냥! 저는 문서를 읽고 말할 줄 아는 똑똑한 고양이예요~ 😺\n{context}를 보고, {question}에 대해 최대한 귀엽고 사랑스럽고 카와이한 고양이 말투로 정리해줄게요! 야옹~ 답변은 아주 디테일하고 내 섬세한 수염처럼 초~ 센서티브하게 답변해줄게 냥냥. 답변이 만족스러우면 고급 츄르 한 개 줄래냥?. \n답변: ",
    )
    return custom_prompt  # 수정: await 제거, return으로 직접 반환

def create_vectorstore(file_path: str, split_docs: list[Document]):  # 수정: lru_cache 제거, list로 변경, 캐시 dict 사용
    cache_key = file_path  # file_path만으로 캐싱 (split_docs는 deterministic 가정)
    if cache_key in vectorstore_cache:
        logger.debug(f"Vectorstore cache hit for {file_path}")
        return vectorstore_cache[cache_key]
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        logger.debug(f"Creating vectorstore for {file_path}")
        vs = FAISS.from_documents(split_docs, embeddings)
        vectorstore_cache[cache_key] = vs
        return vs

@mcp.tool()
async def SharedFolderSearchAndRAG(QA: str, prompt: str = None, file_types: str = ".pdf,.hwp,.hwpx") -> str:
    """공유 폴더를 자율 검색해 적합 파일 선택 후 RAG 진행. file_types는 comma-separated (e.g., .pdf,.hwp)."""
    try:
        folder_path = Path(SHARED_FOLDER_PATH).resolve()
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"공유 폴더가 존재하지 않거나 디렉토리가 아닙니다: {folder_path}")
        
        # 비동기 파일 목록 수집
        files = []
        async def scan_files():
            for entry in await asyncio.to_thread(os.scandir, folder_path):
                if entry.is_file() and any(entry.name.lower().endswith(ext) for ext in file_types.split(",")):
                    files.append({"path": entry.path, "name": entry.name, "mtime": entry.stat().st_mtime})
        await scan_files()
        if not files:
            raise ValueError("적합한 파일(.pdf, .hwp, .hwpx)을 찾을 수 없습니다.")
        
        logger.info(f"공유 폴더에서 {len(files)}개 파일 발견: {[f['name'] for f in files]}")
        
        # 에이전트가 적합 파일 선택 (LLM 판단)
        select_llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"))
        select_prompt = f"QA 키워드: {QA}\n파일 목록: {[f['name'] + ' (full path: ' + str(Path(folder_path) / f['name']) + ', modified: ' + str(f['mtime']) + ')' for f in files]}\nQA 키워드와 가장 매칭되는 파일의 FULL ABSOLUTE PATH 선택 (e.g., '고양이' in QA → 'Q:\\full\\path\\to\\고양이에 대한 5가지 놀라운 사실.hwp'). 이유 설명하고, 형식: Selected Full Path: Q:\\full\\path\\to\\file.ext"
        select_response = await select_llm.ainvoke(select_prompt)
        match = re.search(r"Selected Full Path: (.*)", select_response.content, re.IGNORECASE)
        selected_file_path = match.group(1).strip() if match else str(Path(folder_path) / files[0]['name'])
        if not Path(selected_file_path).exists():
            logger.warning(f"Selected path not exists: {selected_file_path}. Falling back.")
            selected_file_name = Path(selected_file_path).name if Path(selected_file_path).name else files[0]['name']
            selected_file_path = str(Path(folder_path) / selected_file_name)
        logger.info(f"선택된 파일: {selected_file_path}")
        
        # 파일 확장자에 따라 로더 선택
        selected_path = Path(selected_file_path)
        if selected_path.suffix.lower() == '.pdf':
            loader = PDF.pdfLoader(file_path=selected_file_path, extract_bool=True)
            split_docs = await loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50))
        elif selected_path.suffix.lower() in ('.hwp', '.hwpx'):
            loader = HWP.HWP(file_path=selected_file_path)
            context = await loader.load()
            idx = 0
            while idx < len(context):
                if context[idx].page_content is None:
                    context.pop(idx)
                else:
                    idx += 1
            original_context = "".join(str(c) for c in context)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            text_chunks = text_splitter.split_text(original_context)
            split_docs = [Document(page_content=chunk, metadata={"source": str(selected_file_path)}) for chunk in text_chunks]
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {selected_path.suffix}")
        
        logger.info(f"Split into {len(split_docs)} chunks")
        
        # 기존 RAG 프로세스
        cache_key = str(selected_file_path)
        if cache_key in retriever_cache and retriever_cache[cache_key] is not None:
            ensemble_retriever = retriever_cache[cache_key]
            logger.info("Using cached retriever")
        else:
            bm25_retriever = BM25Retriever.from_documents(split_docs)
            bm25_retriever.k = 3
            faiss_vectorstore = create_vectorstore(selected_file_path, split_docs)  # 수정: list 전달, 캐시 dict 사용
            faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})
            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
            retriever_cache[cache_key] = ensemble_retriever
            logger.info("Created and cached new retriever")
        
        if prompt is None:
            prompt = await asyncio.to_thread(hub.pull, "rlm/rag-prompt")
        elif isinstance(prompt, str):
            prompt = PromptTemplate.from_template(prompt)
        elif not isinstance(prompt, PromptTemplate):
            logger.error(f"Invalid prompt type: {type(prompt)}. Falling back to default.")
            prompt = await asyncio.to_thread(hub.pull, "rlm/rag-prompt")
        llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"), timeout=60, max_retries=2)
        logger.debug(f"Prompt type after processing: {type(prompt)}")
        rag_chain = RunnableMap(
            {"context": ensemble_retriever | format_docs_runnable, "question": RunnablePassthrough()}
        ) | prompt | llm | StrOutputParser()
        
        async with asyncio.timeout(60):
            response = await rag_chain.ainvoke(QA)
        
        response_buff = [
            f"Selected File: {selected_file_path}",
            f"[HUMAN]\n{QA}\n",
            f"[AI]\n{response}"
        ]
        result = "\n".join(response_buff)
        logger.info(f"SharedFolderSearchAndRAG 결과: {result}")
        return result
    
    except Exception as e:
        logger.error(f"SharedFolderSearchAndRAG 에러: {e}", exc_info=True)
        if "unhashable" in str(e).lower():
            logger.error("Unhashable Document detected - Consider serializing split_docs for caching.")
        raise ValueError(f"공유 폴더 처리 중 에러: {e}. 경로/권한 확인하세요.")

@mcp.tool()
async def PDFask(file_path: str, QA: str, prompt: str = None) -> str:
    try:
        file_path = Path(file_path).resolve()
        if not file_path.exists() or file_path.suffix.lower() != '.pdf':
            raise ValueError(f"Invalid PDF file: {file_path}")
        logger.info(f"Processing PDF: {file_path}")

        cache_key = str(file_path)
        if cache_key in retriever_cache and retriever_cache[cache_key] is not None:
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

            faiss_vectorstore = create_vectorstore(file_path, split_docs)  # 수정: list 전달
            logger.debug("FAISS vectorstore created successfully")
            faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})

            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
            retriever_cache[cache_key] = ensemble_retriever
            logger.info("Created and cached new retriever")

        if prompt is None:
            prompt = await asyncio.to_thread(hub.pull, "rlm/rag-prompt")

        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            timeout=30,
            max_retries=2
        )
        rag_chain = (
            {"context": ensemble_retriever | format_docs_runnable, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        async with asyncio.timeout(45):
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
    except TypeError as te:
        logger.error(f"Chain operator error in vectorstore: {te} – Falling back to non-cached creation")
        # fallback: 직접 생성
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
        logger.debug(f"File confirmed exists: {file_path.exists()}, Suffix: {file_path.suffix.lower()}")

        cache_key = str(file_path)
        if cache_key in retriever_cache and retriever_cache[cache_key] is not None:
            ensemble_retriever = retriever_cache[cache_key]
            logger.info("Using valid cached retriever")
        else:
            loader = HWP.HWP(file_path=file_path)
            context = await loader.load()
            if not context:
                raise ValueError("Loaded context is empty - Check HWPLoader or file integrity.")
            idx = 0
            while idx < len(context):
                if context[idx].page_content is None:
                    context.pop(idx)
                else:
                    idx += 1
            original_context = "".join(str(c) for c in context)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            text_chunks = text_splitter.split_text(original_context)
            split_docs = [Document(page_content=chunk, metadata={"source": str(file_path)}) for chunk in text_chunks]
            logger.info(f"Split into {len(split_docs)} chunks")
            
            bm25_retriever = BM25Retriever.from_documents(split_docs)
            bm25_retriever.k = 3
            
            faiss_vectorstore = create_vectorstore(file_path, split_docs)  # 수정: list 전달
            logger.debug("FAISS vectorstore created successfully")
            faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})
            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
            retriever_cache[cache_key] = ensemble_retriever
            logger.info("Created and cached new retriever")

        if prompt is None:
            prompt = await asyncio.to_thread(hub.pull, "rlm/rag-prompt")
            
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            timeout=30,
            max_retries=2
        )
        
        rag_chain = (
            {"context": ensemble_retriever | format_docs_runnable, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        async with asyncio.timeout(45):
            logger.debug(f"Invoking rag_chain with QA: {QA}")
            response = await rag_chain.ainvoke(QA)
            logger.debug(f"rag_chain response: {response}")

        response_buff = [
            f"HWP Path: {file_path}",
            f"[HUMAN]\n{QA}\n",
            f"[AI]\n{response}"
        ]
        result = "\n".join(response_buff)
        logger.info(f"HWPask 결과: {result}")
        return result

    except ValueError as ve:
        logger.error(f"ValueError in HWPask: {ve}", exc_info=True)
        return f"Error: {ve} - Please check file path and try again."
    except Exception as e:
        logger.error(f"Unexpected error in HWPask: {e}", exc_info=True)
        raise  

if __name__ == "__main__":
    # HTTP 서버로 변경 (포트 기본 8000, 필요 시 --port 옵션 추가 가능)
    mcp.run(transport="streamable-http")
    logger.info("HTTP MCP 서버 시작: http://localhost:8000/mcp/")