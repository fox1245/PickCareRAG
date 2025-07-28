import init

class pdfLoader():
    def __init__(self, file_path, extract_bool = True):
        self.path = file_path
        self.extract_bool = extract_bool
        
        

    def load(self):
        self.loader = init.PyPDFLoader(self.path, extract_images= self.extract_bool)
        self.docs = self.loader.load()
        print(type(self.docs))
        return self.docs

