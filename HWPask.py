import init
import platform
if platform.system() == "Windows":
    import pythoncom
    import win32com.client
    from pyhwpx import Hwp
elif platform.system() == 'Linux':
    print("한글, word 로더 사용불가")
if init.platform.system() == "Windows": 
    import hwpLoader as HL
    
def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

def HWPask(file_path, QA, model = "gpt-5", prompt = None, k = 3):
    loader = HL.hwpLoader(file_path= file_path)
    context = loader.load2()
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
    