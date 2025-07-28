import init
class WebBaseLoader():
    def __init__(self, url):
        self.web_paths = [url,]
    
    def parse(self, attrArgs: dict):
        loader = init.WebBaseLoader(
            web_paths = self.web_paths,
            bs_kwargs = dict(
                parse_only = init.bs4.SoupStrainer(
                    "div",
                    attrs=attrArgs,

                )
            ),
        )
        docs : list = loader.load()
        return docs
    
    

