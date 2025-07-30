import init

class StreamCallBack(init.BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(token, end = "", flush = True)
    