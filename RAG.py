import init
import WebBaseLoader as WB
import pdfLoader as PDF
import jsonLoader as JL
import pptLoader as PL
import testClass as TC
import csvLoader as CL
import CLIP_RAG as CLIP
import hwpLoader as HL
# API 키 정보 로드
init.load_dotenv()

#init.logging.langsmith("Pickcare-RAG")

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)



def WebLoad(url, model, QA, attrs, html_class, prompt = None):
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


def PDFask(file_path, model, QA, prompt = None, k = 3):
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


def JSONask(file_path, jq_schema,  QA ,model = "gpt-4o", prompt = None, k = 3, text_content = False):
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
        
        
def HWPask(file_path, QA, model = "gpt-4o", prompt = None, k = 3):
    loader = HL.hwpLoader(file_path= file_path)
    context = loader.load()
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


def RAG_RunnableWithMessageHistory(file_path, ask : dict,  session_id: dict , model = "gpt-4o-mini",temp = 0,  k=3, chunk_size = 1000, chunk_overlap = 50):
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
    
    
    #모델 생성
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



        
        


    



        
#set INCLUDE=%INCLUDE%;"A:\vcpkg\installed\x64-windows\include\leptonica"
#set LIB=%LIB%;"A:\vcpkg\installed\x64-windows\lib"  




if __name__ == "__main__":
    # TC.TestClass.test_webBase()
    # TC.TestClass.test_webBase2()
    # TC.TestClass.testJSON()
    # TC.TestClass.testPDF()
    # TC.TestClass.testPPT()
    # loader = CL.csvLoader(file_path = "data/titanic.csv")
    # docs = loader.load()
    # for elem in docs:
    #     print(elem.page_content)
    # QA = "삼성 가우스에 대해 설명해주세요"
    # file = "data/SPRI_AI_Brief_2023년12월호_F.pdf"
    # file4 = "Tensorrt_demos 빌드 방법 정리.pdf"
    file3 = "data/people.json"
    # pdfQuery = PDFask(model = "gpt-4o-mini", QA = QA, file_path = file)
    # for elem in pdfQuery:
    #     print(elem)
    # global store
    # store = {}
    # session_id = {'session_id' : 'rag123'}
    # ask = {'system': '당신은 Question-Answering 챗봇입니다. 주어진 질문에 대한 답변을 제공해주세요.', 'question': '주어진 자료에서 핵심 사항을 요약해서 노래로 만들어 주세요'}
    # response = simpleChatWithHistory(ask)
    # response = RAG_RunnableWithMessageHistory(file_path = file, ask = ask, session_id= session_id)
    # print(response)
    # import unstructured.partition.pdf
    # print("설치 성공!")
    
    
    # clip = CLIP.CLIP(4000, 0, file4, fpath = "data/")
    # chain = clip.load()
    # query = "다음 문서가 말하고자하는 내용에 대해서 쉬운 용어로 설명하세요.  그리고 상세히 내용을 분석하세요. 그리고 주어진 이미지들에 대해서 설명하세요. 그리고 당신이 정리한 모든 내용을 일본어로 번역하세요"
    # response = chain.invoke(query, limit = 6)
    # print(response)
    # test_json = JL.jsonLoader(file_path = file3, jq_schema=".[].phoneNumbers", text_content= False)
    # text_splitter = init.RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
    # print(test_json.load())
    
    # json_response = JSONask(file_path = file3, jq_schema= ".[].phoneNumbers", QA = "다음 문서를 분석하여 요약하세요")
    # print(json_response)
    
    # hwp = HL.hwpLoader(r"Q:\Coding\PickCareRAG\디지털 정부혁신 추진계획.hwp")
    # docs = hwp.load()
    # print(docs)
    hwp_response = HWPask(file_path= r"Q:\Coding\PickCareRAG\재미있는 일본어 단어암기.hwp", QA = "다음 문서에서 핵심 단어나 사회적, 경제적, 실용적으로 중요한 의미의 내용을 간추려서 단어장 만들어 주세요")
    for elem in hwp_response:
        print(elem, end = "", flush = True)

