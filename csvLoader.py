import init

class csvLoader():
    def __init__(self, file_path):
        self.file_path = file_path
    def load(self):
        loader = init.CSVLoader(self.file_path)
        docs = loader.load()
        return docs
    