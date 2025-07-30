from dotenv import load_dotenv
from xai_sdk import Client
import os
import io
from PIL import Image


load_dotenv()
client = Client(api_key=os.getenv("GROK_API_KEY"))


def create_image(model_name = "grok-2-image" , prompt = "A cat in a tree", file_name = "output.png"):
  response = client.image.sample(
    model=model_name,#"grok-2-image",
    prompt=prompt,
    image_format="base64"
  )

  image_bytes = response.image

  # Pillow로 이미지 로드 (메모리에서 처리)
  img = Image.open(io.BytesIO(image_bytes))

  file_name = file_name
  img.save(f"{file_name}", format ="PNG")
  print(f"이미지가 무손실 PNG 포맷으로 {file_name} 파일에 저장되었습니다.")
  
  
  
if __name__ == "__main__":
  create_image(prompt = "A gorgeous male stagbeetle", file_name = "output_images/stag_beetle.png")
