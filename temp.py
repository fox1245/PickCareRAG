@mcp.tool()
async def HWPask(file_path: str, QA : str, prompt: str = None) -> str:
    """확장자가 *.hwp이거나 *.hwpx인 파일을 읽을 수 있다."""
    try:
        file_path = Path(file_path).resolve()
        if not file_path.exists() or file_path.suffix.lower() not in ('.hwp', '.hwpx'):
            raise ValueError(f"Invalid HWP or HWPX file: {file_path}")
        logger.info(f"Processing HWP: {file_path}")
        cache_key = str(file_path)
        if cache_key in retriever_cache:
            ensemble_retriever = retriever_cache[cache_key]
            logger.info("Using cached retriever")
        else:
            loader = HWP.HWP(file_path=file_path)
            
                
            context = loader.load()
            idx = 0
            for c in context:
                if c.page_content == None:
                    context.pop(idx)
            original_context = ""
            for c in context:
                original_context += str(c)
            
            
            context = original_context
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
            text_chunks = text_splitter.split_text(context)
            #문자열 리스트를 Document 객체 리스트로 변환
            split_docs = [Document(page_content = chunk, metadata = {"source":file_path}) for chunk in text_chunks]
            
                

            
            # split_docs = await loader.load_and_split(
            #     text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            # )
            
            
            #대략 여기서
            
            logger.info(f"Split into {len(split_docs)} chunks")

            #대략 여기서


            bm25_retriever = BM25Retriever.from_documents(split_docs)
            bm25_retriever.k = 3

            # 수정: create_vectorstore 호출 시 split_docs 직접 넘김 (캐싱 키는 file_path만)
            # lru_cache가 file_path로 캐싱되지만, 실제 생성은 split_docs 사용
            create_vectorstore(file_path)  # 캐싱 체크 (하지만 실제 생성은 아래)
            faiss_vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings(model="text-embedding-3-large"))  # 직접 생성, lru_cache는 상태 확인용으로
            logger.debug("FAISS vectorstore created successfully")  # 성공 로그 추가
            faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})

            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
            retriever_cache[cache_key] = ensemble_retriever
            logger.info("Created and cached new retriever")
                
            
        if prompt is None:
            prompt = await asyncio.to_thread(hub.pull, "rlm/rag-prompt")
                
                
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            timeout=30,  # API 호출 타임아웃 30초
            max_retries=2  # 재시도 2회
        )
        
        rag_chain = (
            {"context": ensemble_retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        async with asyncio.timeout(45):  # RAG 체인 타임아웃 45초 (기존 유지)
            logger.debug(f"Invoking rag_chain with QA: {QA}")
            response = await rag_chain.ainvoke(QA)
            logger.debug(f"rag_chain response: {response}")

        response_buff = [
            f"PDF Path: {file_path}",
            f"[HUMAN]\n{QA}\n",
            f"[AI]\n{response}"
        ]
        result = "\n".join(response_buff)
        logger.info(f"HWPask 결과: {result}")
        return result

    except asyncio.TimeoutError:
        logger.error(f"Timeout in PDFask for {file_path}")
        raise ValueError("PDF processing or LLM call timed out")
    except TypeError as te:  # 추가: unhashable 에러 fallback
        logger.error(f"Hash error in vectorstore: {te} – Falling back to non-cached creation")
        # fallback: 캐싱 없이 직접 생성 (필요 시)
        faiss_vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings(model="text-embedding-3-large"))
    except Exception as e:
        logger.error(f"Error in PDFask: {e}", exc_info=True)
        raise
        
        
        
        
@mcp.tool()
async def HWPask(file_path: str, QA: str, prompt: str = None) -> str:
    """확장자가 *.hwp이거나 *.hwpx인 파일을 읽을 수 있다."""
    try:
        file_path = Path(file_path).resolve()
        if not file_path.exists() or file_path.suffix.lower() not in ('.hwp', '.hwpx'):
            raise ValueError(f"Invalid HWP or HWPX file: {file_path}")
        logger.info(f"Processing HWP: {file_path}")
        logger.debug(f"File exists: {file_path.exists()}, Suffix: {file_path.suffix.lower()}")  # 디버그 추가: 경로 확인
        
        cache_key = str(file_path)
        if cache_key in retriever_cache:
            ensemble_retriever = retriever_cache[cache_key]
            logger.info("Using cached retriever")
        else:
            loader = HWP.HWP(file_path=file_path)
            context = await loader.load()  # 수정: await 추가! (async 메서드 호출)
            logger.debug(f"Loaded context type: {type(context)}, Length: {len(context) if context else 0}")  # 디버그: 로드 확인
            
            idx = 0
            while idx < len(context):  # 안전하게 pop 위해 while 사용 (pop 시 인덱스 변함)
                if context[idx].page_content is None:
                    context.pop(idx)
                else:
                    idx += 1
            
            original_context = "".join(str(c) for c in context)  # 효율적 join
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            text_chunks = text_splitter.split_text(original_context)
            split_docs = [Document(page_content=chunk, metadata={"source": str(file_path)}) for chunk in text_chunks]
            
            logger.info(f"Split into {len(split_docs)} chunks")
            
            bm25_retriever = BM25Retriever.from_documents(split_docs)
            bm25_retriever.k = 3
            
            create_vectorstore(file_path)  # 기존 유지
            faiss_vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings(model="text-embedding-3-large"))
            logger.debug("FAISS vectorstore created successfully")
            faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 3})
            
            ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
            retriever_cache[cache_key] = ensemble_retriever
            logger.info("Created and cached new retriever")
        
        # 나머지 코드 (prompt, llm, rag_chain) 기존 유지
        # ...
    except Exception as e:
        logger.error(f"Error in HWPask: {e}", exc_info=True)  # 에러 로그 강화 (HWPask로 변경)
        raise