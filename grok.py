import init
init.load_dotenv()



class GrokRAG():
    
    def __init__(self):
        self.client = init.Client(api_key=init.os.getenv("GROK_API_KEY"))
        self.chat = init.ChatXAI( xai_api_key = init.os.getenv('GROK_API_KEY'), model = "grok-4",)
        

    def create_image(self, model_name = "grok-2-image" , prompt = "A cat in a tree", file_name = "output.png"):
        
        self.response = self.client.image.sample(
            model=model_name,#"grok-2-image",
            prompt=prompt,
            image_format="base64"
        )

        self.image_bytes = self.response.image

        # Pillow로 이미지 로드 (메모리에서 처리)
        self.img = init.Image.open(init.io.BytesIO(self.image_bytes))

        self.file_name = file_name
        self.img.save(f"{file_name}", format ="PNG")
        print(f"이미지가 무손실 PNG 포맷으로 {file_name} 파일에 저장되었습니다.")
    
    



    


  
# if __name__ == "__main__":
#     g = GrokRAG()
#     g.create_image(prompt = "A tiger jumping up to the mountains", file_name = "output_images/tiger.png")
#     for m in g.chat.stream("Tell me fun thing to do in NYC"):
#         print(m.content, end = "", flush = True)
