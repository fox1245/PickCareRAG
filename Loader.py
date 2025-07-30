import init
from abc import ABC, abstractclassmethod

class Loader(ABC):
    @abstractclassmethod
    def load():
        """문서를 로드하여 사용할 수 있는 객체로 반환"""
        pass
    