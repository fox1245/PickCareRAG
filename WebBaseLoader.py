import init
class WebBaseLoader():
    def __init__(self, url):
        self.web_paths = [url,]
    
    def parse(self):
        loader = init.WebBaseLoader(
            web_paths = self.web_paths,
            bs_kwargs = dict(
                parse_only = init.bs4.SoupStrainer(
                    "div",
                    attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},

                )
            ),
        )
        docs : list = loader.load()
        return docs
    
    

