# math_server.py
from mcp.server.fastmcp import FastMCP
import pdfLoader as PDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


mcp = FastMCP("Simple RAG")


def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)



@mcp.tool()
async def PDFask(file_path, QA, prompt):
    try:
        loader = PDF.pdfLoader(file_path = file_path, extract_bool= True)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
        split_docs = loader.load_and_split(text_splitter = text_splitter)
        bm25_retriever = BM25Retriever.from_documents(split_docs)   
        bm25_retriever.k = 3
        faiss_vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings(model="text-embedding-3-large"))
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs = {"k" : 3})

        
        #앙상블 리트리버를 초기화
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weight = [0.5, 0.5])
        
        
        #프롬프트 생성
        if prompt == None:
            prompt = hub.pull("rlm/rag-prompt") #프롬프트
        
        llm = ChatOpenAI(model_name = "gpt-5-mini")
        rag_chain = (
            {"context" : ensemble_retriever | format_docs, "question" : RunnablePassthrough()}
            |prompt
            |llm
            |StrOutputParser()
        )
        
            #체인 실행
        question = QA
        response = rag_chain.invoke(QA)
        response_buff = list()
        response_buff.append(f"PDF Path: {file_path}")
        response_buff.append(f"[HUMAN]\n{QA}\n")
        response_buff.append(f"[AI]\n{response}")

        
        
        await response_buff
        
    except Exception as PDFaskException:
        print(f"{PDFaskException}")
    
    
    

if __name__ == "__main__":
    mcp.run(transport="stdio")