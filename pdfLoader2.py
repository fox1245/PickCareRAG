import asyncio
import logging
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from typing import List, Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class pdfLoader:
    def __init__(self, file_path: str, extract_bool: bool = True):
        """PDF 로더 초기화"""
        self.path = Path(file_path).resolve()  # 절대 경로로 변환
        self.extract_bool = extract_bool
        self.loader = PyMuPDFLoader(self.path, extract_images=extract_bool)
        
    async def load(self) -> List[Document]:
        """PDF 파일을 비동기적으로 로드"""
        try:
            if not self.path.exists() or self.path.suffix.lower() != '.pdf':
                raise ValueError(f"Invalid PDF file: {self.path}")
            logger.info(f"Loading PDF: {self.path}")
            docs = await asyncio.to_thread(self.loader.load)  # 동기 호출을 비동기화
            logger.info(f"Loaded {len(docs)} documents from {self.path}")
            return docs
        except Exception as e:
            logger.error(f"Error loading PDF {self.path}: {e}", exc_info=True)
            raise

    async def load_and_split(self, text_splitter) -> List[Document]:
        """PDF 파일을 비동기적으로 로드하고 텍스트 분할"""
        try:
            if not self.path.exists() or self.path.suffix.lower() != '.pdf':
                raise ValueError(f"Invalid PDF file: {self.path}")
            logger.info(f"Loading and splitting PDF: {self.path}")
            split_docs = await asyncio.to_thread(self.loader.load_and_split, text_splitter=text_splitter)
            logger.info(f"Split into {len(split_docs)} chunks from {self.path}")
            return split_docs
        except Exception as e:
            logger.error(f"Error splitting PDF {self.path}: {e}", exc_info=True)
            raise