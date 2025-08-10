import init
from Loader import Loader
from typing import List
from langchain_core.documents import Document

class docxLoader(Loader):
    def __init__(self, file_path):
        self.file_path = file_path
        self.loader = init.Docx2txtLoader(f"{file_path}") #문서 로더 초기화
        
    def load(self) -> List[Document]:
        try:
            documents = self.loader.load()
            if not documents:
                raise ValueError(f"{self.file_path}로 부터 문서를 로드 실패")
            return documents
        except FileNotFoundError:
            raise FileNotFoundError(f"파일이 없습니다. {self.file_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading DOCX: {str(e)}")
        
        
        
        
        