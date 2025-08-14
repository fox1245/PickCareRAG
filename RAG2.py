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
import glob
import os
from langchain_core.caches import SQLiteCache
from langchain.globals import set_llm_cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))  # 디스크에 캐시 저장


global store
store = {}




# API 키 정보 로드
init.load_dotenv()

#init.logging.langsmith("Pickcare-RAG")

# tqdm 객체를 전역 변수로 선언 (callback에서 공유)
progress_bar = None

def callback(step: int, steps: int, time: float):
    print("Completed step: {} of {}".format(step, steps))
    
def callback2(step: int, steps: int, time: float):
    global progress_bar
    if progress_bar is None:
        progress_bar = init.tqdm(total=steps, desc="Generating Image")  # 초기화
    progress_bar.update(1)  # 진행 상황 업데이트
    if step == steps:  # 완료 시 클리어
        progress_bar.close()

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)


@tool
def WebLoad(url, model, QA, attrs, html_class, prompt = None):
    """사용자 쿼리에 맞는 문서를 web에서 검색합니다."""
    parseMan = WB.WebBaseLoader(url)
    docs = parseMan.load(attrArgs = attrs, klass = html_class)

    #문서분할
    text_splitter = init.RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)

    splits = text_splitter.split_documents(docs)

    #벡터스토어 생성
    vectorstore = init.FAISS.from_documents(documents = splits, embedding = init.OpenAIEmbeddings(model="text-embedding-3-large"))
    
    #모델 생성
    llm = init.ChatOpenAI(model_name = model)

    #검색(search)
    retriever = vectorstore.as_retriever()
    
    if prompt == None:
        prompt = init.hub.pull("rlm/rag-prompt") #프롬프트
    
    #체인 생성
    rag_chain = (
    {"context": retriever | format_docs, "question": init.RunnablePassthrough()}
    | prompt
    | llm
    | init.StrOutputParser()
    )
    #체인 실행
    question = QA
    response  = rag_chain.invoke(question)
    
    response_buff = list()
    response_buff.append(f"URL: {url}")
    response_buff.append(f"문서의 수: {len(docs)}")
    response_buff.append(f"[HUMAN]\n{question}\n")
    response_buff.append(f"[AI]\n{response}")
    return response_buff

@tool
def PDFask(file_path, model, QA, prompt = None, k = 3):
    """PDF를 로드하고 RAG로 처리하여 PDF 문서의 내용에 대해 쿼리하기 적합한 함수입니다."""
    loader = PDF.pdfLoader(file_path= file_path, extract_bool=True)
    
    text_splitter = init.RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
    
    split_docs = loader.load_and_split(text_splitter = text_splitter)
    #print(type(split_docs))
    vectorstore = init.FAISS.from_documents(documents= split_docs, embedding = init.OpenAIEmbeddings(model="text-embedding-3-large"))
    
    
    bm25_retriever = init.BM25Retriever.from_documents(split_docs)    
    bm25_retriever.k = k
    faiss_vectorstore = init.FAISS.from_documents(split_docs, init.OpenAIEmbeddings(model="text-embedding-3-large"))
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs = {"k" : k})
    
    
    #앙상블 리트리버를 초기화
    ensemble_retiever = init.EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weight = [0.5, 0.5]
    )
    
    #프롬프트 생성
    if prompt == None:
        prompt = init.hub.pull("rlm/rag-prompt") #프롬프트
    
    llm = init.ChatOpenAI(model_name = model)
    
    rag_chain = (
        {"context" : ensemble_retiever | format_docs, "question" : init.RunnablePassthrough()}
        |prompt
        |llm
        |init.StrOutputParser()
    )
    
    response = rag_chain.invoke(QA)
    response_buff = list()
    response_buff.append(f"PDF Path: {file_path}")
    response_buff.append(f"[HUMAN]\n{QA}\n")
    response_buff.append(f"[AI]\n{response}")
    return response_buff

@tool
def JSONask(file_path, jq_schema,  QA , model = "gpt-5", prompt = None, k = 3, text_content = False):
    """JSON을 로드하고 사용자의 쿼리 내용에 맞도록 처리하여 결과를 리턴하는 함수입니다."""
    loader = JL.jsonLoader(file_path = file_path, jq_schema= jq_schema, text_content = text_content)
    
    text_splitter = init.RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
    loader.load()
    split_docs = loader.load_and_split(text_splitter=text_splitter)
    #vectorstore = init.FAISS.from_documents(documents = split_docs, embedding = init.OpenAIEmbeddings(model = "text-embedding-3-large"))
    
    bm25_retriever = init.BM25Retriever.from_documents(split_docs)    
    bm25_retriever.k = k
    faiss_vectorstore = init.FAISS.from_documents(split_docs, init.OpenAIEmbeddings(model="text-embedding-3-large"))
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs = {"k" : k})
    #앙상블 리트리버를 초기화
    ensemble_retiever = init.EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weight = [0.5, 0.5]
    )
    
    if prompt == None:
        prompt = init.hub.pull("rlm/rag-prompt") #프롬프트
    
    llm = init.ChatOpenAI(model_name = model)
    
    rag_chain = (
        {"context" : ensemble_retiever | format_docs, "question" : init.RunnablePassthrough()}
        |prompt
        |llm
        |init.StrOutputParser()
    )
    
    response = rag_chain.invoke(QA)
    response_buff = list()
    response_buff.append(f"JSON Path: {file_path}")
    response_buff.append(f"[HUMAN]\n{QA}\n")
    response_buff.append(f"[AI]\n{response}")
    return response_buff
        
        

    
    


def promptMaker(Prompt : dict):
    system_prompt = Prompt['system']
    question = Prompt['question']
    prompt = init.ChatPromptTemplate.from_messages(
        [
            ("system",
             f"{system_prompt}"
            ),
            init.MessagesPlaceholder(variable_name= "chat_history"),
            ("human" , "#Question:\n{question}"),  # 사용자 입력을 변수로 사용
        ]
        
    )
    return prompt


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    print(f"[대화 세션ID]: {session_ids}")
    if session_ids not in store:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        store[session_ids] = init.ChatMessageHistory()
    return store[session_ids]  # 해당 세션 ID에 대한 세션 기록 반환
    

def simpleChatWithHistory(ask):
    prompt = promptMaker(Prompt = ask)
    llm = init.ChatOpenAI()
    chain = prompt | llm | init.StrOutputParser()
    #세션 기록을 저장할 딕셔너리
    
    chain_with_history = init.RunnableWithMessageHistory(
        chain, 
        get_session_history, #세션 기록을 가져오는 함수
        input_messages_key = "question", #사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key= "chat_history", #기록 메시지의 키
    )
    
    
    response = chain_with_history.invoke(
        #질문 입력
        {"question" : ask["question"]},
        config={"configurable": {"session_id": "abc123"}},
         
    )
    return response

@tool
def RAG_RunnableWithMessageHistory(file_path, ask : dict,  session_id: dict , model = "gpt-5-mini",temp = 0,  k=3, chunk_size = 1000, chunk_overlap = 50):
    """
    사용자와의 대화 내용을 기억하면서 채팅하고 싶을 때 사용하는 함수
    """
    #문서 로드
    loader = PDF.pdfLoader(file_path=file_path, extract_bool= True)
    docs = loader.load()
    #문서 분할
    text_splitter = init.RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    split_documents = text_splitter.split_documents(docs)
    #임베딩 생성
    embeddings = init.OpenAIEmbeddings()
    
    #DB생성 및 저장
    vectorstore = init.FAISS.from_documents(documents= split_documents, embedding= embeddings)
    
    #검색기 (Retriever) 생성
    retriever = vectorstore.as_retriever(search_kwargs = {"k" : k})
    
    #프롬프트 생성

    
    system = ask['system']
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

            #Answer:"""
    )
    
    
    if "gpt-5" in model:
    #모델 생성
        llm = init.ChatOpenAI(model_name = model, temperature= None)
    else:
        llm = init.ChatOpenAI(model_name = model, temperature= temp)
    
    
    #체인 생성
    chain = (
        {
            "context" : init.itemgetter("question") | retriever,
            "question" : init.itemgetter("question"),
            "chat_history" : init.itemgetter("chat_history"),
        }
        | prompt
        | llm
        | init.StrOutputParser()
    )
    
    global store
    store = dict()
    
    
    rag_with_history = init.RunnableWithMessageHistory(
        chain, 
        get_session_history, #세션 기록을 가져오는 함수
        input_messages_key= "question", #사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key= "chat_history",  #기록 메시지의 키
        
    )
    
    response = rag_with_history.invoke(
        #질문 입력
        {"question" : ask["question"]},
        config = {"configurable" : {"session_id" : session_id["session_id"]}},
    )
    
    return response

def prompt_maker():
    custom_prompt = init.PromptTemplate(
    input_variables=["context", "question"],
    template="냥! 저는 문서를 읽고 말할 줄 아는 똑똑한 고양이예요~ 😺\n{context}를 보고, {question}에 대해 최대한 귀엽고 사랑스럽고 카와이한 고양이 말투로 정리해줄게요! 야옹~ 답변은 아주 디테일하고 내 섬세한 수염처럼 초~ 센서티브하게 답변해줄게 냥냥. 답변이 만족스러우면 고급 츄르 한 개 줄래냥?. \n답변: ",
    )
    return custom_prompt
    
        
        

@tool
def HWPask2(file_path, QA, model = "gpt-5", prompt = None, k = 3):
    """
    Use this function for files with extensions ".hwp" or ".hwpx"
    """
    loader = HWP.HWP(file_path= file_path)
    context = loader.load()
    idx = 0
    for c in context:
        if c.page_content == None:
            context.pop(idx)
    original_context = ""
    for c in context:
        original_context += str(c)
    
    context = original_context
            

    text_splitter = init.RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
    text_chunks = text_splitter.split_text(context)
    # 문자열 리스트를 Document 객체 리스트로 변환
    split_docs = [init.Document(page_content=chunk, metadata={"source": file_path}) for chunk in text_chunks]
    #vectorstore = init.FAISS.from_documents(documents = split_docs, embedding = init.OpenAIEmbeddings(model = "text-embedding-3-large"))
    
    bm25_retriever = init.BM25Retriever.from_documents(split_docs)    
    bm25_retriever.k = k
    faiss_vectorstore = init.FAISS.from_documents(split_docs, init.OpenAIEmbeddings(model="text-embedding-3-large"))
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs = {"k" : k})
    #앙상블 리트리버를 초기화
    ensemble_retiever = init.EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever], weight = [0.5, 0.5]
    )
    
    if prompt == None:
        prompt = init.hub.pull("rlm/rag-prompt") #프롬프트
    
    prompt += "\n answer in korean"
    
    
    llm = init.ChatOpenAI(model_name = model)
    
    rag_chain = (
        {"context" : ensemble_retiever | format_docs, "question" : init.RunnablePassthrough()}
        #{"context" : init.RunnableLambda(format_docs) | ensemble_retiever, "question" : init.RunnablePassthrough()}
        |prompt
        |llm
        |init.StrOutputParser()
    )
    
    response = rag_chain.invoke(QA)
    response_buff = list()
    response_buff.append(f"HWP Path: {file_path}")
    response_buff.append(f"[HUMAN]\n{QA}\n")
    response_buff.append(f"[AI]\n{response}")
    return response_buff
    
    

# LangGraph 에이전트
def DynamicRAGAgent(query: str, model: str = "gpt-4o-mini"):
    """사용자 쿼리에 따라 문서를 동적으로 검색하고 RAG로 처리합니다."""
    try:
        llm = init.ChatOpenAI(model_name=model, cache=True)
        tools = [WebLoad, PDFask, JSONask, HWPask2, RAG_RunnableWithMessageHistory]
        agent = create_react_agent(llm, tools)
        
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
        return response["messages"][-1].content
    except Exception as e:
        return f"Error in DynamicRAGAgent: {str(e)}"



if __name__ == "__main__":
    # 캐시 디렉토리 생성
    if not os.path.exists("cache"):
        os.makedirs("cache")
    user_query = "AI 관련 PDF 문서에서 최신 트렌드 알려줘"
    result = DynamicRAGAgent(user_query)
    print(result)
    
    
    




    

            

