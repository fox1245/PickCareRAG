import init
from Loader import Loader


class WebBaseLoader(Loader):
    def __init__(self, url):
        self.web_paths = [url,]
    
    def load(self, attrArgs: dict, klass : str):
        loader = init.WebBaseLoader(
            web_paths = self.web_paths,
            bs_kwargs = dict(
                parse_only = init.bs4.SoupStrainer(
                    klass,
                    attrs=attrArgs,

                )
            ),
        )
        docs : list = loader.load()
        return docs
    
    

