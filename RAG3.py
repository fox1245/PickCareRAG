import init
import WebBaseLoader as WB
import pdfLoader as PDF
import jsonLoader as JL
import pptLoader as PL
import csvLoader as CL
import CLIP_RAG as CLIP
import docxLoader_old as WL
from test_grok import create_image
import HWP
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.cache import SQLiteCache
from langchain.globals import set_llm_cache
import glob
import os
from pathlib import Path
import logging
from retry import retry
import fitz
import unicodedata
import hashlib


# 캐싱 초기화
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

global store
store = {}

# API 키 정보 로드
init.load_dotenv()

# tqdm 객체를 전역 변수로 선언
progress_bar = None

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = init.ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))
    
def callback2(step: int, steps: int, time: float):
    global progress_bar
    if progress_bar is None:
        progress_bar = init.tqdm(total=steps, desc="Generating Image")
    progress_bar.update(1)
    if step == steps:
        progress_bar.close()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Tool 1: 동적 문서 검색
@tool
def search_documents(query: str) -> list:
    """사용자 쿼리에 맞는 문서 파일을 검색합니다. PDF, JSON, HWP 지원."""
    try:
        base_dir = "Q:/Coding/PickCareRAG/data"  # 실제 문서 디렉토리
        candidates = (
            glob.glob(os.path.join(base_dir, f"*{query}*.pdf")) +
            glob.glob(os.path.join(base_dir, f"*{query}*.json")) +
            glob.glob(os.path.join(base_dir, f"*{query}*.hwp"))
        )
        if not candidates:
            return ["No matching documents found."]
        return candidates[:3]
    except Exception as e:
        return [f"Error searching documents: {str(e)}"]

# Tool 2: Web 문서 로드 및 쿼리
@tool
def WebLoad(url: str, qa: str, model: str = "gpt-4o-mini", attrs: dict = None, html_class: str = None, prompt: str = None) -> str:
    """사용자 쿼리에 맞는 문서를 웹에서 검색하고 RAG로 처리합니다."""
    try:
        parseMan = WB.WebBaseLoader(url, user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)")
        docs = parseMan.load(attrArgs=attrs, klass=html_class)
        text_splitter = init.RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)
        
        # FAISS 캐싱
        cache_path = f"cache/{url.replace('/', '_')}.faiss"
        if Path(cache_path).exists():
            vectorstore = init.FAISS.load_local(cache_path, embeddings=init.OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization=True)
        else:
            vectorstore = init.FAISS.from_documents(documents=splits, embedding=init.OpenAIEmbeddings(model="text-embedding-3-large"))
            vectorstore.save_local(cache_path)
        
        llm = init.ChatOpenAI(model_name=model, cache=True)
        retriever = vectorstore.as_retriever()
        if prompt is None:
            prompt = init.hub.pull("rlm/rag-prompt") + "\n answer in korean"
        
        rag_chain = (
            {"context": retriever | format_docs, "question": init.RunnablePassthrough()}
            | prompt
            | llm
            | init.StrOutputParser()
        )
        
        response = rag_chain.invoke(qa)
        response_buff = [f"URL: {url}", f"문서의 수: {len(docs)}", f"[HUMAN]\n{qa}\n", f"[AI]\n{response}"]
        return "\n".join(response_buff)
    except Exception as e:
        return f"Error in WebLoad: {str(e)}"

# Tool 3: PDF 문서 로드 및 쿼리


@tool
@retry(tries=3, delay=2)  # 끈질기게 3번 재시도
def PDFask(file_path: str, qa: str, model: str = "gpt-4o-mini", prompt: str = None, k: int = 3) -> str:
    """PDF 문서를 로드하고 RAG로 쿼리합니다."""
    logging.basicConfig(level=logging.INFO)
    try:
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return f"Error: PDF 파일 '{file_path}'이 존재하지 않습니다."

        # 캐시 디렉토리 자동 생성 (문제 해결 포인트 1)
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            logging.info(f"캐시 디렉토리 생성: {cache_dir}")

        # 파일명 안전화: 한글을 ASCII로 변환 (문제 해결 포인트 2)
        safe_filename = unicodedata.normalize('NFKD', Path(file_path).stem).encode('ascii', 'ignore').decode('ascii')
        safe_filename = safe_filename.replace(' ', '_')  # 띄어쓰기 등 처리
        if not safe_filename:  # 만약 빈 문자열이면 해시 사용
            safe_filename = hashlib.md5(file_path.encode()).hexdigest()
        cache_path = f"{cache_dir}/{safe_filename}.faiss"
        logging.info(f"안전한 캐시 경로: {cache_path}")

        # 로더 부분 (이전과 동일, fallback 추가)
        try:
            loader = PDF.pdfLoader(file_path=file_path, extract_bool=True)
            docs = loader.load()
        except Exception as load_err:
            logging.warning(f"기본 로더 실패: {str(load_err)}. PyMuPDF fallback.")
            docs = []
            with fitz.open(file_path) as pdf_doc:
                for page in pdf_doc:
                    text = page.get_text()
                    docs.append(init.Document(page_content=text, metadata={"source": file_path, "page": page.number}))

        if not docs:
            return "PDF 텍스트 추출 실패."

        text_splitter = init.RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)
        
        # FAISS 캐싱 (안정화)
        vectorstore = None
        try:
            if Path(cache_path).exists():
                vectorstore = init.FAISS.load_local(cache_path, embeddings=init.OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization=True)
        except Exception as cache_err:
            logging.warning(f"캐시 로드 실패: {str(cache_err)}. 새로 생성합니다.")
        
        if vectorstore is None:
            try:
                vectorstore = init.FAISS.from_documents(documents=split_docs, embedding=init.OpenAIEmbeddings(model="text-embedding-3-large"))
                vectorstore.save_local(cache_path)
            except Exception as save_err:
                logging.error(f"FAISS 저장 실패: {str(save_err)}. Chroma fallback 사용.")
                # Fallback: Chroma 벡터스토어 (pip install chromadb)
                from langchain_community.vectorstores import Chroma
                chroma_cache = f"{cache_dir}/{safe_filename}_chroma"
                vectorstore = Chroma.from_documents(documents=split_docs, embedding=init.OpenAIEmbeddings(model="text-embedding-3-large"), persist_directory=chroma_cache)
                vectorstore.persist()  # 저장

        # 나머지 RAG 체인 (이전과 동일)
        bm25_retriever = init.BM25Retriever.from_documents(split_docs)
        bm25_retriever.k = k
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        ensemble_retriever = init.EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
        
        llm = init.ChatOpenAI(model_name=model, cache=True)
        if prompt is None:
            prompt = init.hub.pull("rlm/rag-prompt") + "\n answer in korean"
        
        rag_chain = (
            {"context": ensemble_retriever | format_docs, "question": init.RunnablePassthrough()}
            | prompt
            | llm
            | init.StrOutputParser()
        )
        
        response = rag_chain.invoke(qa)
        response_buff = [f"PDF Path: {file_path}", f"[HUMAN]\n{qa}\n", f"[AI]\n{response}"]
        return "\n".join(response_buff)
    except Exception as e:
        logging.error(f"PDFask 전체 오류: {str(e)}")
        return f"Error in PDFask: {str(e)}. 로그 확인하고, 경로를 영어로 변경해 보세요."
# Tool 4: JSON 문서 로드 및 쿼리
@tool
def JSONask(file_path: str, qa: str, jq_schema: str = ".data[]", model: str = "gpt-4o-mini", prompt: str = None, k: int = 3, text_content: bool = False) -> str:
    """JSON 문서를 로드하고 RAG로 쿼리합니다."""
    try:
        loader = JL.jsonLoader(file_path=file_path, jq_schema=jq_schema, text_content=text_content)
        text_splitter = init.RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        loader.load()
        split_docs = loader.load_and_split(text_splitter=text_splitter)
        
        # FAISS 캐싱
        cache_path = f"cache/{Path(file_path).name}.faiss"
        if Path(cache_path).exists():
            vectorstore = init.FAISS.load_local(cache_path, embeddings=init.OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization=True)
        else:
            vectorstore = init.FAISS.from_documents(documents=split_docs, embedding=init.OpenAIEmbeddings(model="text-embedding-3-large"))
            vectorstore.save_local(cache_path)
        
        bm25_retriever = init.BM25Retriever.from_documents(split_docs)
        bm25_retriever.k = k
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        ensemble_retriever = init.EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
        
        llm = init.ChatOpenAI(model_name=model, cache=True)
        if prompt is None:
            prompt = init.hub.pull("rlm/rag-prompt") + "\n answer in korean"
        
        rag_chain = (
            {"context": ensemble_retriever | format_docs, "question": init.RunnablePassthrough()}
            | prompt
            | llm
            | init.StrOutputParser()
        )
        
        response = rag_chain.invoke(qa)
        response_buff = [f"JSON Path: {file_path}", f"[HUMAN]\n{qa}\n", f"[AI]\n{response}"]
        return "\n".join(response_buff)
    except Exception as e:
        return f"Error in JSONask: {str(e)}"

# Tool 5: HWP 문서 로드 및 쿼리
@tool
def HWPask2(file_path: str, qa: str, model: str = "gpt-4o-mini", prompt: str = None, k: int = 3) -> str:
    """HWP 문서를 로드하고 RAG로 쿼리합니다."""
    try:
        loader = HWP.HWP(file_path=file_path)
        context = loader.load()
        context = [c for c in context if c.page_content is not None]
        original_context = "".join(str(c) for c in context)
        
        text_splitter = init.RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        text_chunks = text_splitter.split_text(original_context)
        split_docs = [init.Document(page_content=chunk, metadata={"source": file_path}) for chunk in text_chunks]
        
        # FAISS 캐싱
        cache_path = f"cache/{Path(file_path).name}.faiss"
        if Path(cache_path).exists():
            vectorstore = init.FAISS.load_local(cache_path, embeddings=init.OpenAIEmbeddings(model="text-embedding-3-large"), allow_dangerous_deserialization=True)
        else:
            vectorstore = init.FAISS.from_documents(documents=split_docs, embedding=init.OpenAIEmbeddings(model="text-embedding-3-large"))
            vectorstore.save_local(cache_path)
        
        bm25_retriever = init.BM25Retriever.from_documents(split_docs)
        bm25_retriever.k = k
        faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        ensemble_retriever = init.EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
        
        llm = init.ChatOpenAI(model_name=model, cache=True)
        if prompt is None:
            prompt = init.hub.pull("rlm/rag-prompt") + "\n answer in korean"
        
        rag_chain = (
            {"context": ensemble_retriever | format_docs, "question": init.RunnablePassthrough()}
            | prompt
            | llm
            | init.StrOutputParser()
        )
        
        response = rag_chain.invoke(qa)
        response_buff = [f"HWP Path: {file_path}", f"[HUMAN]\n{qa}\n", f"[AI]\n{response}"]
        return "\n".join(response_buff)
    except Exception as e:
        return f"Error in HWPask2: {str(e)}"

# Tool 6: RAG with Message History
@tool
def RAG_RunnableWithMessageHistory(file_path: str, question: str, session_id: str, model: str = "gpt-4o-mini", temp: float = 0, k: int = 3, chunk_size: int = 1000, chunk_overlap: int = 50) -> str:
    """대화 기록을 유지하며 PDF 문서를 RAG로 쿼리합니다."""
    try:
        loader = PDF.pdfLoader(file_path=file_path, extract_bool=True)
        docs = loader.load()
        text_splitter = init.RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_documents = text_splitter.split_documents(docs)
        
        # FAISS 캐싱
        cache_path = f"cache/{Path(file_path).name}.faiss"
        if Path(cache_path).exists():
            vectorstore = init.FAISS.load_local(cache_path, embeddings=init.OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        else:
            vectorstore = init.FAISS.from_documents(documents=split_documents, embedding=init.OpenAIEmbeddings())
            vectorstore.save_local(cache_path)
        
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        prompt = init.PromptTemplate.from_template(
            """
            You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Answer in Korean.

            #Previous Chat History:
            {chat_history}

            #Question: 
            {question} 

            #Context: 
            {context} 

            #Answer:
            """
        )
        
        llm = init.ChatOpenAI(model_name=model, temperature=temp, cache=True)
        chain = (
            {
                "context": init.itemgetter("question") | retriever,
                "question": init.itemgetter("question"),
                "chat_history": init.itemgetter("chat_history"),
            }
            | prompt
            | llm
            | init.StrOutputParser()
        )
        
        rag_with_history = init.RunnableWithMessageHistory(
            chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="chat_history",
        )
        
        response = rag_with_history.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}},
        )
        
        return response
    except Exception as e:
        return f"Error in RAG_RunnableWithMessageHistory: {str(e)}"

def prompt_maker():
    custom_prompt = init.PromptTemplate(
        input_variables=["context", "question"],
        template="냥! 저는 문서를 읽고 말할 줄 아는 똑똑한 고양이예요~ 😺\n{context}를 보고, {question}에 대해 최대한 귀엽고 사랑스럽고 카와이한 고양이 말투로 정리해줄게요! 야옹~ 답변은 아주 디테일하고 내 섬세한 수염처럼 초~ 센서티브하게 답변해줄게 냥냥. 답변이 만족스러우면 고급 츄르 한 개 줄래냥?. \n답변: ",
    )
    return custom_prompt

# LangGraph 에이전트
def DynamicRAGAgent(query: str, model: str = "gpt-4o-mini"):
    try:
        llm = init.ChatOpenAI(model_name=model, cache=True)
        tools = [search_documents, WebLoad, PDFask, JSONask, HWPask2, RAG_RunnableWithMessageHistory]
        agent = create_react_agent(llm, tools)
        
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
        final_content = response["messages"][-1].content
        
        # 에러 발생 시 사용자에게 피드백
        if "Error" in final_content:
            return final_content + "\n\n추가 팁: 파일 경로를 절대 경로로 확인하거나, PDF 라이브러리를 업데이트하세요. 재시도 해보세요!"
        
        return final_content
    except Exception as e:
        return f"에이전트 오류: {str(e)}. 로그를 확인하고 재실행하세요."

if __name__ == "__main__":
    # 캐시 디렉토리 생성
    if not os.path.exists("cache"):
        os.makedirs("cache")
    
    # 테스트 쿼리
    file = r"Q:\Coding\PickCareRAG\data\Tensorrt_demos.pdf"
    user_query = f"search_documents 도구를 사용해서 확인 가능한 모든 문서의 이름을 대라 그리고 문서의 저자의 성씨가 최씨인 문서를 찾아라"
    result = DynamicRAGAgent(user_query)
    print(result)