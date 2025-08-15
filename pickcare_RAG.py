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
import json
from jsonLoader import jsonLoader
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Optional, List
from langchain_core.messages import HumanMessage  # langgraph를 위한 import 추가

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

def flatten_json(data, prefix=''):  # 추가: 플래튼 헬퍼 함수 (재귀, 인터넷 없이 구현)
    """JSON을 평탄화하는 헬퍼. 중첩을 'key.subkey'로 변환."""
    flat = {}
    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, (dict, list)):
                flat.update(flatten_json(value, new_key))
            else:
                flat[new_key] = value
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_key = f"{prefix}[{i}]"
            flat.update(flatten_json(item, new_key))
    else:
        flat[prefix] = data
    return flat

@mcp.tool()
async def JSONask(file_path: str, QA: str, jq_schema: str = None, prompt: str = None, k: int = 3, text_content: bool = False, model: str = "gpt-5-mini") -> str:
    """JSON 파일을 처리하는 도구. jq_schema 옵션(기본 None: 플래튼 모드). RAG로 QA 답변."""
    try:
        file_path = Path(file_path).resolve()
        if not file_path.exists() or file_path.suffix.lower() != '.json':
            folder_path = Path(SHARED_FOLDER_PATH).resolve()
            matching_files = [f for f in folder_path.glob('*.json') if 'pet' in f.name.lower() or '펫' in f.name.lower()]
            if matching_files:
                file_path = matching_files[0]  # 첫 번째 매칭 파일 사용
                logger.warning(f"Fallback to shared JSON: {file_path}")
            else:
                raise ValueError(f"Invalid JSON file: {file_path}. 공유 폴더에도 없음.")
        logger.info(f"Processing JSON: {file_path}")

        cache_key = str(file_path)
        if cache_key in retriever_cache and retriever_cache[cache_key] is not None:
            ensemble_retriever = retriever_cache[cache_key]
            logger.info("Using cached retriever")
        else:
            # jq_schema 없으면 플래튼 모드
            if jq_schema is None:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                flat_data = flatten_json(data)
                # 플래튼 결과를 문자열로 (key: value 형식)
                json_text = "\n".join(f"{k}: {v}" for k, v in flat_data.items())
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
                split_docs = text_splitter.split_text(json_text)
                split_docs = [Document(page_content=chunk, metadata={"source": str(file_path)}) for chunk in split_docs]
            else:
                # 기존 JSONLoader 사용
                loader = jsonLoader(file_path=str(file_path), jq_schema=jq_schema, text_content=text_content)
                split_docs = await loader.load_and_split(text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50))

            logger.info(f"Split into {len(split_docs)} chunks")

            bm25_retriever = BM25Retriever.from_documents(split_docs)
            bm25_retriever.k = k

            faiss_vectorstore = create_vectorstore(str(file_path), split_docs)
            faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": k})

            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
            retriever_cache[cache_key] = ensemble_retriever
            logger.info("Created and cached new retriever")

        if prompt is None:
            prompt = await asyncio.to_thread(hub.pull, "rlm/rag-prompt")

        llm = ChatOpenAI(model_name=model, timeout=30, max_retries=2)
        rag_chain = (
            {"context": ensemble_retriever | format_docs_runnable, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        async with asyncio.timeout(45):
            response = await rag_chain.ainvoke(QA)

        response_buff = [
            f"JSON Path: {file_path}",
            f"[HUMAN]\n{QA}\n",
            f"[AI]\n{response}"
        ]
        result = "\n".join(response_buff)
        logger.info(f"JSONask 결과: {result}")
        return result

    except Exception as e:
        logger.error(f"Error in JSONask: {e}", exc_info=True)
        raise ValueError(f"JSON 처리 중 에러: {e}")

@mcp.tool()
async def GetCurrentDate() -> str:
    """현재 시스템 날짜를 반환하는 도구. 날짜 관련 질문 시 사용. 반환 형식: YYYY-MM-DD (예: 2025-08-15)."""
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"GetCurrentDate 호출: {current_date}")
        return current_date
    except Exception as e:
        logger.error(f"GetCurrentDate 에러: {e}", exc_info=True)
        raise ValueError(f"날짜 가져오기 중 에러: {e}")

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
async def SharedFolderSearchAndRAG(QA: str, prompt: str = None, file_types: str = ".pdf,.hwp,.hwpx,.json", history: str = None) -> str:
    """공유 폴더를 자율 검색해 적합 파일 선택 후 RAG 진행. file_types는 comma-separated (e.g., .pdf,.hwp, .json)."""
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
            raise ValueError("적합한 파일(.pdf, .hwp)을 찾을 수 없습니다.")
        
        logger.info(f"공유 폴더에서 {len(files)}개 파일 발견: {[f['name'] for f in files]}")
        
        # 수정: history가 있으면 QA에 붙임 (대화 맥락 반영으로 파일 선택 정확도 ↑)
        if history:
            QA = f"{history}\n{QA}"
        
        # 에이전트가 적합 파일 선택 (LLM 판단)
        select_llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-5-mini"))
        # 파일 선택 프롬프트 수정: history 반영 (이전 히스토리를 명시적으로 포함해 LLM이 맥락 이해)
        select_prompt = f"이전 히스토리: {history or '없음'}\nQA 키워드: {QA}\n파일 목록: {[f['name'] + ' (full path: ' + str(Path(folder_path) / f['name']) + ', modified: ' + str(f['mtime']) + ')' for f in files]}\nQA 키워드와 가장 매칭되는 파일의 FULL ABSOLUTE PATH 선택 (e.g., '고양이' in QA → 'Q:\\full\\path\\to\\고양이에 대한 5가지 놀라운 사실.hwp'). 이유 설명하고, 형식: Selected Full Path: Q:\\full\\path\\to\\file.ext"
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
        elif selected_path.suffix.lower() in ('.hwp'):
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
            
        elif selected_path.suffix.lower() == '.json':
            # JSON 처리: jq_schema None으로 플래튼 모드 사용 (JSONask 로직 재활용)
            with open(selected_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            flat_data = flatten_json(data)
            json_text = "\n".join(f"{k}: {v}" for k, v in flat_data.items())
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            text_chunks = text_splitter.split_text(json_text)
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
        llm = ChatOpenAI(model_name=os.getenv("OPENAI_MODEL_NAME", "gpt-5-mini"), timeout=60, max_retries=2)
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
            model_name="gpt-5-mini",
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
            model_name="gpt-5-mini",
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

app = FastAPI(title="RAG MCP API", description="RESTful API for RAG with MCP tools")
# app = FastAPI(...) 바로 아래에 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용 모든 오리진 허용. 프로덕션 시 ["http://your-frontend-domain.com"]로 제한
    allow_credentials=True,
    allow_methods=["*"],  # OPTIONS, POST 등 모든 메서드 허용
    allow_headers=["*"],  # Content-Type 등 모든 헤더 허용
)


# Pydantic 모델: JSON 페이로드 정의 (쿼리 입력 유효성 검사)
class QueryRequest(BaseModel):
    query: str  # 사용자 쿼리 (필수)
    file_path: Optional[str] = None  # 옵션: 파일 경로 (PDFask 등에 사용)
    tool: Optional[str] = None  # 옵션: 호출할 MCP 도구 이름 (e.g., "PDFask")
    history: Optional[List[dict]] = None  # 옵션: 대화 히스토리 (client처럼 요약 전달)

class QueryResponse(BaseModel):
    result: str  # RAG 결과
    source: Optional[str] = None  # 사용된 파일/도구 소스
    timestamp: str  # 처리 시간

# RESTful 엔드포인트: POST /api/rag/query (주 엔드포인트)
@app.post("/api/rag/query", response_model=QueryResponse)
async def rag_query(request: QueryRequest = Body(...)):
    try:
        logger.info(f"Received query: {request.query}")
        # MCP 호출 로직: 쿼리에 따라 도구 선택 (기존 agent처럼)
        # 예: SharedFolderSearchAndRAG 도구 호출 (history 포함)
        if request.tool == "SharedFolderSearchAndRAG":
            result = await SharedFolderSearchAndRAG(
                QA=request.query,
                history="\n".join([f"{msg['role']}: {msg['content']}" for msg in request.history or []]) if request.history else None
            )
        elif request.tool == "PDFask":
            if not request.file_path:
                raise HTTPException(status_code=400, detail="file_path required for PDFask")
            result = await PDFask(file_path=request.file_path, QA=request.query)
        elif request.tool == "HWPask":
            if not request.file_path:
                raise HTTPException(status_code=400, detail="file_path required for HWPask")
            result = await HWPask(file_path=request.file_path, QA=request.query)
        elif request.tool == "JSONask":
            if not request.file_path:
                raise HTTPException(status_code=400, detail="file_path required for JSONask")
            result = await JSONask(file_path=request.file_path, QA=request.query)
        # ... (다른 도구들 추가: GetCurrentDate, prompt_maker 등 필요 시 if-elif 확장)
        else:
            # 기본: 에이전트처럼 MCP 전체 호출 (PleaseBE5_client.py 스타일)
            from langgraph.prebuilt import create_react_agent  # import 추가 (런타임에 불러서 메모리 절약)
            tools = await mcp.get_tools()  # MCP 도구들
            agent = create_react_agent(model="openai:gpt-5-mini", tools=tools)
            input_message = {"messages": [HumanMessage(content=request.query)]}
            agent_response = await agent.ainvoke(input_message)
            result = agent_response["messages"][-1].content

        return QueryResponse(
            result=result,
            source=request.file_path or "Shared Folder",
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error in rag_query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 추가 엔드포인트: GET /api/health (헬스 체크)
@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "1.0"}

# MCP 도구 목록 노출: GET /api/tools
@app.get("/api/tools")
async def get_tools():
    tools = await mcp.get_tools()
    return {"tools": [tool.name for tool in tools]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)  # 로컬 테스트