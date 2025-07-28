import init
import WebBaseLoader as WB
# API 키 정보 로드
init.load_dotenv()

#init.logging.langsmith("Pickcare-RAG")

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)



def WebLoad(url, model, QA, attrs, html_class):
    parseMan = WB.WebBaseLoader(url)
    docs = parseMan.parse(attrArgs = attrs, klass = html_class)

    #문서분할
    text_splitter = init.RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)

    splits = text_splitter.split_documents(docs)

    #벡터스토어 생성
    vectorstore = init.FAISS.from_documents(documents = splits, embedding = init.OpenAIEmbeddings())

    #검색(search)
    retriever = vectorstore.as_retriever()


    prompt = init.hub.pull("rlm/rag-prompt")

    #모델 생성
    llm = init.ChatOpenAI(model_name = model)

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


curious_man = "부영그룹의 출산 장려 정책에 대해 설명해주세요"
web_res = WebLoad(url = "https://n.news.naver.com/article/437/0000378416", model = "gpt-4o-mini", QA = curious_man, attrs= {"class": ["newsct_article _article_body", "media_end_head_title"]}, html_class = 'div')
for elem in web_res:
    print(elem)


curious_man2 = "주어진 문서를 분석하세요"
web_res2 = WebLoad(url = "https://www.bbc.com/news/business-68092814", model = "gpt-4o-mini", QA = curious_man2, attrs = {"id": ["main-content"]}, html_class= 'main')
for elem in web_res2:
    print(elem)
