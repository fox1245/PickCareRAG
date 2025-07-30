import init
from Loader import Loader

class pptLoader(Loader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        loader = init.UnstructuredPowerPointLoader(self.file_path)

        docs = loader.load()
        return docs
