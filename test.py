import init
import WebBaseLoader as WB
import pdfLoader as PDF
import jsonLoader as JL
import pptLoader as PL
import testClass as TC
import csvLoader as CL
# API 키 정보 로드
init.load_dotenv()

#init.logging.langsmith("Pickcare-RAG")

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)



def WebLoad(url, model, QA, attrs, html_class):
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


def PDFLoad(file_path, model, QA):
    loader = PDF.pdfLoader(file_path= file_path, extract_bool=True)
    
    



if __name__ == "__main__":
    TC.TestClass.test_webBase()
    TC.TestClass.test_webBase2()
    #TC.TestClass.testJSON()
    #TC.TestClass.testPDF()
    #TC.TestClass.testPPT()



# loader = CL.csvLoader(file_path = "data/titanic.csv")
# docs = loader.load()
# for elem in docs:
#     print(elem.page_content)
