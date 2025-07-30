import init
import WebBaseLoader as WB
import pdfLoader as PDF
import jsonLoader as JL
import pptLoader as PL
from RAG import WebLoad



class TestClass():
    def __init__(self):
        pass
    def test_webBase():    
        curious_man = "부영그룹의 출산 장려 정책에 대해 설명해주세요. 그리고 한국어로 답변해주셔야 합니다."
        web_res = WebLoad(url = "https://n.news.naver.com/article/437/0000378416", model = "gpt-4o-mini", QA = curious_man, attrs= {"class": ["newsct_article _article_body", "media_end_head_title"]}, html_class = 'div')
        for elem in web_res:
            print(elem)

    def test_webBase2():
        curious_man2 = "주어진 문서를 분석하세요. 그리고 한국어로 답변해주셔야 합니다."
        web_res2 = WebLoad(url = "https://www.bbc.com/news/business-68092814", model = "gpt-4o-mini", QA = curious_man2, attrs = {"id": ["main-content"]}, html_class= 'main')
        for elem in web_res2:
            print(elem)

    def testPDF():
        FILE_PATH = "./data/SPRI_AI_Brief_2023년12월호_F.pdf"

        PDF_context = PDF.pdfLoader(FILE_PATH).load()
        for elem in PDF_context:
            print(elem.page_content)


    def testJSON():
        load_json = JL.jsonLoader(
            file_path = "data/people.json",
            jq_schema= ".[].phoneNumbers",
            text_content = False,
        )
        print(load_json.load())
    
    
    def testPPT():
        ppt_file_path = "./data/sample-ppt.pptx"
        ppt_load = PL.pptLoader(ppt_file_path).load()

        for elem in ppt_load:
            print(elem)

        