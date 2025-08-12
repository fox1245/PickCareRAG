import init
import WebBaseLoader as WB
import pdfLoader as PDF
import jsonLoader as JL
import pptLoader as PL
import csvLoader as CL
import CLIP_RAG as CLIP
import testClass as TC
if init.platform.system() == "Windows": 
    import hwpLoader as HL
import docxLoader_old as WL
from test_grok import create_image
from trellis import create_3d_from_image
import HWP
import RAG
from dotenv import load_dotenv
from langchain_teddynote import logging
load_dotenv()

logging.langsmith("Main Test")



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
    # file4 = r"Tensorrt_demos.pdf"
    # file3 = "data/people.json"
    # pdfQuery = RAG.PDFask(model = "gpt-5-mini", QA = QA, file_path = file)
    # for elem in pdfQuery:
    #     print(elem)
    # global store
    # store = {}
    # session_id = {'session_id' : 'rag123'}
    # ask = {'system': '당신은 Question-Answering 챗봇입니다. 주어진 질문에 대한 답변을 제공해주세요.', 'question': '주어진 자료에서 핵심 사항을 요약해서 노래로 만들어 주세요'}
    # response = RAG.simpleChatWithHistory(ask)
    # response = RAG.RAG_RunnableWithMessageHistory(file_path = file, ask = ask, session_id= session_id)
    # print(response)
    # import unstructured.partition.pdf
    # print("설치 성공!")
    
    # docx_loader = WL.docxLoader(file_path="data/sample-word-document.docx")
    # docx_doc = docx_loader.load()
    # print("확인")
    # print(docx_doc)
    
    
    # clip = CLIP.CLIP(4000, 0, rf"{file4}", fpath = "data/")
    # chain = clip.load()
    # query = "다음 문서가 말하고자하는 내용에 대해서 쉬운 용어로 설명하세요.  그리고 상세히 내용을 분석하세요. 그리고 주어진 이미지들에 대해서 설명하세요. 그리고 당신이 정리한 모든 내용을 일본어로 번역하세요"
    # response = chain.invoke(query, limit = 6)
    # print(response)
    # test_json = JL.jsonLoader(file_path = file3, jq_schema=".[].phoneNumbers", text_content= False)
    # text_splitter = init.RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
    # print(test_json.load())
    
    # json_response = RAG.JSONask(file_path = file3, jq_schema= ".[].phoneNumbers", QA = "다음 문서를 분석하여 요약하세요")
    # print(json_response)
    # custom_prompt = RAG.prompt_maker()
    # if init.platform.system() == "Windows": 
    #     hwp = HL.hwpLoader(r"Q:\Coding\PickCareRAG\디지털 정부혁신 추진계획.hwp")
    #     docs = hwp.load2()
    #     print(docs)
        
    #     hwp_response = RAG.HWPask(file_path= r"Q:\Coding\PickCareRAG\디지털 정부혁신 추진계획.hwp",prompt = custom_prompt,  QA = "해당 문서를 공부하라고 잔소리만 해대서 집을 뛰쳐나가서 방랑하는 한참 질풍노도의 시기 속에 놓여진 박테리아도 이해할 수 있을 만큼 쉽게 정리하세요")
    #     for elem in hwp_response:
    #         print(elem, end = "", flush = True)
    
    # prompt = init.hub.pull("rlm/rag-prompt")
    # print(prompt)
    # if init.platform.system() == "Windows": 
    #     print("윈도우")
    #     pdf_response = RAG.PDFask(file_path=r"Q:\Coding\PickCareRAG\data\Tensorrt_demos.pdf", model = "gpt-5",  QA = "해당 문서를 길가던 참새도 이해할 수 있을 만큼 쉽게 정리하세요", prompt=custom_prompt)
    # elif init.platform.system() == "Linux":
    #     print("리눅스")
    #     pdf_response = RAG.PDFask(file_path=r"/mnt/q/Coding/PickCareRAG/data/Tensorrt_demos.pdf", model = "gpt-5",  QA = "해당 문서를 충치에 걸린 다람쥐도 이해할 수 있을 만큼 쉽게 정리하세요", prompt=custom_prompt)
    # for elem in pdf_response:
    #     print(elem, end = "", flush = True)
    
    # RAG.create_diffusion_image("cute_cat", "very very cute, lovely, precious kitty staring at me")
        
    
    # #create_image(prompt = "A photo of a cute kitten in Kawhi asking for a snack. An animation from the 1980s in Japan", file_name = "output_images/lovely_cat.png")
    
    # #create_3d_from_image(file_path= "data/doggum.jpg", output_file = "cutecat.mp4")
    
    # hwp_response = RAG.HWPask2(file_path= r"Q:\Coding\PickCareRAG\디지털 정부혁신 추진계획.hwp",prompt = custom_prompt,  QA = "해당 문서를 공부하라고 잔소리만 해대서 집을 뛰쳐나가서 방랑하는 한참 질풍노도의 시기 속에 놓여진 박테리아도 이해할 수 있을 만큼 쉽게 정리하세요")
    # for elem in hwp_response:
    #     print(elem)
    
    QA = "삼성 가우스에 대해 설명해주세요"
    file = "data/SPRI_AI_Brief_2023년12월호_F.pdf"
    pdfQuery = RAG.PDFask(model = "gpt-5-mini", QA = QA, file_path = file)
    res = ""
    for elem in pdfQuery:
        res += elem
        
    print(res)
        
    