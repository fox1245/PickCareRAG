from langchain_community.document_loaders import PyMuPDFLoader
from Loader import Loader

class pdfLoader(Loader):
    def __init__(self, file_path, extract_bool = True):
        self.path = file_path
        self.extract_bool = extract_bool
        self.loader = PyMuPDFLoader(self.path, extract_images= self.extract_bool)
        
        

    def load(self):
        self.docs = self.loader.load()
        print(type(self.docs))
        return self.docs

    def load_and_split(self, text_splitter):
        try:
            split_docs =  self.loader.load_and_split(text_splitter = text_splitter)
            return split_docs
        except Exception as e:
            print(e)

            
    
