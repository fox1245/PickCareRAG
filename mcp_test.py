from mcp.server.fastmcp import FastMCP
import init
import RAG
import HWP
import asyncio




mcp = FastMCP("Test")



def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

import asyncio  # 이미 import 있음


@mcp.tool()
async def HWPask2(file_path: str, QA: str, model="gpt-5", prompt= RAG.prompt_maker(), k=3):  # async def로 변경
    
    """
    Use this function for files with extensions '.hwp' or '.hwpx'
    """
    
    try:
        # Sync 로더 부분을 executor로 offload (지연 방지)
        def sync_load():
            loader = HWP.HWP(file_path=file_path)
            context = loader.load()
            idx = 0
            while idx < len(context):  # for 대신 while로 안전
                if context[idx].page_content is None:
                    context.pop(idx)
                else:
                    idx += 1
            original_context = "".join(str(c) for c in context)
            return original_context

        context = await asyncio.get_running_loop().run_in_executor(None, sync_load)  # offload

        text_splitter = init.RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        text_chunks = text_splitter.split_text(context)
        split_docs = [init.Document(page_content=chunk, metadata={"source": file_path}) for chunk in text_chunks]

        bm25_retriever = init.BM25Retriever.from_documents(split_docs)
        bm25_retriever.k = k
        faiss_vectorstore = init.FAISS.from_documents(split_docs, init.OpenAIEmbeddings(model="text-embedding-3-large"))
        faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": k})
        ensemble_retriever = init.EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]  # weight → weights 오타 수정?
        )

        if prompt is None:
            prompt = init.hub.pull("rlm/rag-prompt")
        prompt += "\n answer in korean"

        llm = init.ChatOpenAI(model_name=model)

        rag_chain = (
            {"context": ensemble_retriever | format_docs, "question": init.RunnablePassthrough()}
            | prompt
            | llm
            | init.StrOutputParser()
        )

        response = await rag_chain.ainvoke(QA)  # 직접 await, 타임아웃은 client 측에서
        response_string = f"HWP Path: {file_path}\n[HUMAN]\n{QA}\n[AI]\n{response}"
        return response_string
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [f"HWP error: {str(e)}"]  # str(3) → str(e)로 수정 (오타?)



@mcp.tool()
def add(a : int, b : int ) -> int:
    """Add two numbers"""
    return a + b

@mcp.tool()
def multiply(a : int , b : int) -> int:
    """Multiplay two numbers"""
    return a * b

@mcp.tool()
def PDFask_custom(QA: str) -> str:
    """If you read the PDF and are asking about the contents of the PDF, use this function"""
    try:
        custom_prompt = RAG.prompt_maker()
        file = r"Q:\Coding\PickCareRAG\data\SPRI_AI_Brief_2023년12월호_F.pdf"
        print(f"Loading PDF: {file}")  # 추가: 로딩 로그
        pdf_response = RAG.PDFask(file_path=file, model="gpt-5", QA=QA, prompt=custom_prompt, k=3)
        res = ""
        for elem in pdf_response:
            res += elem
        return res
    except Exception as e:
        return f"PDF load error: {str(e)}"  # 에러 반환으로 클라이언트에 피드백


@mcp.tool()
def JSONask_custom(QA: str):
    """I use this function when I am asked to analyze JSON documents without knowing what JSON it is."""
    try:
        file3 = r"Q:\Coding\PickCareRAG\data\people.json"
        print(f"Loading JSON: {file3}")  # 추가: 로그
        json_response = RAG.JSONask(file_path=file3, jq_schema=".[].phoneNumbers", QA=QA)
        print(json_response)
        return json_response
    except Exception as e:
        return f"JSON load error: {str(e)}"


        
    


    
    

    
    



if __name__ == "__main__":
    mcp.run(transport= "stdio")
    
