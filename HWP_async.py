from Loader import Loader  # Loader가 어디서 오는지 확인 (추정: langchain_core.loaders.BaseLoader 상속)
from langchain_teddynote.document_loaders import HWPLoader
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HWP(Loader):
    def __init__(self, file_path):
        self.file_path = Path(file_path).resolve()  # 절대 경로로 변환
        self.loader = HWPLoader(self.file_path)
        
    async def load(self):
        """HWP파일을 비동기적으로 로드"""
        try:
            if not self.file_path.exists() or self.file_path.suffix.lower() not in ('.hwp', '.hwpx'):
                raise ValueError(f"Invalid HWP or HWPX file: {self.file_path}")
            logger.info(f"Loading HWP: {self.file_path}")  # self.path → self.file_path
            self.docs = await asyncio.to_thread(self.loader.load)  # 메서드 참조: load (괄호 없음)
            logger.info(f"Loaded {len(self.docs)} documents from {self.file_path}")  # self.path → self.file_path
            return self.docs
        except Exception as e:
            logger.error(f"Error loading HWP {self.file_path}: {e}", exc_info=True)  # self.path → self.file_path
            raise