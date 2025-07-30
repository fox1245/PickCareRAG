import init
from Loader import Loader

class jsonLoader(Loader):
    def __init__(self, file_path, jq_schema, text_content):
        self.file_path = file_path
        self.jq_schema = jq_schema
        self.text_content : bool = text_content 

    def load(self):
        loader = init.JSONLoader(
            file_path= self.file_path,
            jq_schema= self.jq_schema,
            text_content= self.text_content
        )

        docs = loader.load()
        pretty_str = init.pformat(docs, indent=2, width=40)
        return pretty_str

