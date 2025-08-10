import init
from Loader import Loader


class HWP(Loader):
    def __init__(self, file_path):
        self.file_path = file_path
        self.loader = init.HWPLoader(self.file_path)
        
    def load(self):       
        self.docs = self.loader.load()
        return self.docs
        