import init


class WebBaseLoader():
    def __init__(self, url):
        self.web_paths = [url,]
    
    def parse(self, attrArgs: dict, klass : str):
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
    
    

