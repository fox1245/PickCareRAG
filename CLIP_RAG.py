import init
import pdfLoader as PDF
import re
from Loader import Loader

def plt_img_base64(img_base64):
    """base64 인코딩된 문자열을 이미지로 표시"""
    # base64 문자열을 소스로 사용하는 HTML img 태그 생성
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # HTML을 렌더링하여 이미지 표시
    init.display(init.HTML(image_html))


def looks_like_base64(sb):
    """문자열이 base64로 보이는지 확인"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None


def is_image_data(b64data):
    """
    base64 데이터가 이미지인지 시작 부분을 보고 확인
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = init.base64.b64decode(b64data)[:8]  # 처음 8바이트를 디코드하여 가져옴
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False

def resize_base64_image(base64_string, size=(128, 128)):
    """
    Base64 문자열로 인코딩된 이미지의 크기 조정
    """
    # Base64 문자열 디코드
    img_data = init.base64.b64decode(base64_string)
    img = init.Image.open(init.io.BytesIO(img_data))

    # 이미지 크기 조정
    resized_img = img.resize(size, init.Image.LANCZOS)

    # 조정된 이미지를 바이트 버퍼에 저장
    buffered = init.io.BytesIO()
    resized_img.save(buffered, format=img.format)

    # 조정된 이미지를 Base64로 인코딩
    return init.base64.b64encode(buffered.getvalue()).decode("utf-8")


def split_image_text_types(docs):
    """
    base64로 인코딩된 이미지와 텍스트 분리
    """
    b64_images = []
    texts = []
    for doc in docs:
        # 문서가 Document 타입인 경우 page_content 추출
        if isinstance(doc, init.Document):
            doc = doc.page_content
        if looks_like_base64(doc) and is_image_data(doc):
            doc = resize_base64_image(doc, size=(1300, 600))
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}

def img_prompt_func(data_dict):
    """
    컨텍스트를 단일 문자열로 결합
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = []

    # 이미지가 있으면 메시지에 추가
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
            messages.append(image_message)

    # 분석을 위한 텍스트 추가
    text_message = {
        "type": "text",
        "text": (
            # "You are financial analyst tasking with providing investment advice.\n"
            # "You will be given a mixed of text, tables, and image(s) usually of charts or graphs.\n"
            # "Use this information to provide investment advice related to the user question. Answer in Korean. Do NOT translate company names.\n"
            "You are an analysis expert who can analyze a given document and give you an appropriate answer.\n"
            "Usually, text, tables, images, etc. of charts or graphs are mixed.\n"
            "Use this information to provide information or details related to your questions. Answer in Korean.\n"
            f"User-provided question: {data_dict['question']}\n\n"
            "Text and / or tables:\n"
            f"{formatted_texts}"
        ),
    }
    messages.append(text_message)
    return [init.HumanMessage(content=messages)]







class CLIP(Loader):
    def __init__(self, chunk_size, chunk_overlap, FILE_PATH, model = "gpt-4o", fpath = "multi-modal/"):
        self.FILE_PATH = FILE_PATH
        PDF_context = PDF.pdfLoader(self.FILE_PATH).load()
        self.texts = ""
        for elem in PDF_context:
            self.texts += '\n'.join(elem.page_content)
                        
        self.model = model
        self.chunk_size = chunk_size,
        self.chunk_overlap = chunk_overlap
        self.fpath = fpath
        #텍스트에 대해 특정 토큰 크기 적용
        self.text_splitter = init.CharacterTextSplitter.from_tiktoken_encoder(chunk_size = 4000, chunk_overlap = 0)

    def load(self):
        joined_texts = " ".join(self.texts) #텍스트 결합
        self.texts_4k_token = self.text_splitter.split_text(joined_texts) #분할 실행
        
        self.text_summaries = self.generate_text_summaries(self.texts_4k_token, summarize_texts = True)
        
        self.img_base64_list, self.image_summaries = self.generate_img_summaries(self.fpath)
        
        print(self.image_summaries)
        
        #요약을 색인화하기 위해 사용할 벡터 저장소
        vectorstore = init.Chroma(collection_name = "sample-rag-multi-modal", embedding_function = init.OpenAIEmbeddings(model="text-embedding-3-large"))
        
        #검색기 생성
        self.retriever_multi_vector_img = self.create_multi_vector_retriever(vectorstore, self.text_summaries, self.texts, self.image_summaries, self.img_base64_list)
        self.chain_multimodel_rag = self.multi_modal_rag_chain(self.retriever_multi_vector_img)
        return self.chain_multimodel_rag
        
    
        
    def generate_text_summaries(self, texts, summarize_texts = False):
        """
        texts: 문자열 리스트
        summarize_texts: 텍스트 요약 여부를 결정
        """
        
        prompt_text = """You are an assistant tasked with summarizing text for retrieval. \
                        These summaries will be embedded and used to retrieve the raw text elements. \
                        Give a concise summary of the text that is well optimized for retrieval. text: {element} """
        prompt = init.ChatPromptTemplate.from_template(prompt_text)
        
        
        #텍스트 요약 체인
        model = init.ChatOpenAI(temperature=0, model = self.model)
        summarize_chain = {"element": lambda x : x} | prompt | model | init.StrOutputParser()
        
        
        #요약을 위한 빈 리스트 초기화
        text_summaries = []
        
        #제공된 텍스트에 대해 요약이 요청되었을 경우 적용
        if texts and summarize_texts:
            text_summaries = summarize_chain.batch(texts, {"max_concurrency": 2})
        elif texts:
            text_summaries = texts
        return text_summaries
    
    def encode_image(self, image_path):
        #이미지 파일을 base64 문자열로 인코딩.
        with open(image_path, "rb") as image_file:
            return init.base64.b64encode(image_file.read()).decode("utf-8")
        
        
    def image_summarize(self, img_base64, prompt):
        #이미지 요약을 생성
        chat = init.ChatOpenAI(model = self.model, max_tokens = 2048)
        
        msg = chat.invoke(
            [
                init.HumanMessage(
                    content = [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                        },
                    ]
                )
            ]
        )
        return msg.content
    
    
    
    def generate_img_summaries(self, path):
        """
    이미지에 대한 요약과 base64 인코딩된 문자열을 생성합니다.
    path: Unstructured에 의해 추출된 .jpg 파일 목록의 경로
    """
        #base64로 인코딩된 이미지를 저장할 리스트
        img_base64_list = []
        
        #이미지 요약을 저장할 리스트
        image_summaries = []
        
        #요약을 위한 프롬프트
        prompt = """You are an assistant tasked with summarizing images for retrieval. \
                    These summaries will be embedded and used to retrieve the raw image. \
                    Give a concise summary of the image that is well optimized for retrieval."""
                    
        #이미지에 적용
        for img_file in sorted(init.os.listdir(path)):
            if img_file.endswith(".jpg"):
                img_path = init.os.path.join(path, img_file)
                base64_image = self.encode_image(img_path)
                img_base64_list.append(base64_image)
                image_summaries.append(self.image_summarize(base64_image, prompt))
                
        return img_base64_list, image_summaries
    
    
    def create_multi_vector_retriever(self, vectorstore, text_summaries, texts, image_summaries, images):
        """
    요약을 색인화하지만 원본 이미지나 텍스트를 반환하는 검색기를 생성합니다.
    """
        #저장 계층 초기화
        store = init.InMemoryStore()
        id_key = "doc_id"
        
        #멀티 벡터 검색기 생성
        retriever = init.MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore = store,
            id_key = id_key,
        )
        
        #문서를 벡터 저장소와 문서 저장소에 추가하는 헬퍼 함수
        def add_documents(retriever, doc_summaries, doc_contents):
            doc_ids = [
                str(init.uuid.uuid4())for _ in doc_contents
            ] #문서 내용마다 고유 ID생성
            summary_docs = [
                init.Document(page_content = s, metadata = {id_key: doc_ids[i]})
                for i, s in enumerate(doc_summaries)
            ]
            
            retriever.vectorstore.add_documents(
                summary_docs
            ) #요약 문서를 벡터 저장소에 추가
            
            retriever.docstore.mset(
                list(zip(doc_ids, doc_contents))
            ) #문서 내용을 문서 저장소에 추가
            
            
        #텍스트, 테이블, 이미지 추가
        if text_summaries:
            add_documents(retriever, text_summaries, texts)
        if image_summaries:
            add_documents(retriever, image_summaries, images)
            
        return retriever
    
    def multi_modal_rag_chain(self,retriever):
        """
        멀티모달 RAG 체인
        """

        # 멀티모달 LLM
        model = init.ChatOpenAI(temperature=0, model="gpt-4o", max_tokens=2048)

        # RAG 파이프라인
        chain = (
            {
                "context": retriever | init.RunnableLambda(split_image_text_types),
                "question": init.RunnablePassthrough(),
            }
            | init.RunnableLambda(img_prompt_func)
            | model
            | init.StrOutputParser()
        )

        return chain
    

    

        
        
        
        
        
        
    

        
        
        