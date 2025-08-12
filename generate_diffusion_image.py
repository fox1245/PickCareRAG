import init
from stable_diffusion_cpp import StableDiffusion

def callback2(step: int, steps: int, time: float):
    global progress_bar
    if progress_bar is None:
        progress_bar = init.tqdm(total=steps, desc="Generating Image")  # 초기화
    progress_bar.update(1)  # 진행 상황 업데이트
    if step == steps:  # 완료 시 클리어
        progress_bar.close()



def create_diffusion_image(file_name, prompt):
    if init.platform.system() == "Windows": 
        stable_diffusion = StableDiffusion(
            model_path=r"D:\Coding\PythonCleanCode\practice_20250710\model\prefectPonyXL_v40.safetensors",
            # wtype="default", # Weight type (e.g. "q8_0", "f16", etc) (The "default" setting is automatically applied and determines the weight type of a model file)
        )
    elif init.platform.system() == "Linux":
        stable_diffusion = StableDiffusion(
            model_path=r"/mnt/d/Coding/PythonCleanCode/practice_20250710/model/prefectPonyXL_v40.safetensors",
            # wtype="default", # Weight type (e.g. "q8_0", "f16", etc) (The "default" setting is automatically applied and determines the weight type of a model file)
        )
        
    output = stable_diffusion.txt_to_img(
        prompt=prompt,
        width=512, # Must be a multiple of 64
        height=512, # Must be a multiple of 64
        progress_callback=callback2,
        # seed=1337, # Uncomment to set a specific seed (use -1 for a random seed)
    )
    output[0].save(f"output_images/{file_name}.png") # Output returned as list of PIL Images
    
    
    image_path = f"output_images/{file_name}.png"
    if init.os.path.exists(image_path):
        img = init.Image.open(image_path)
        
        init.plt.rc('font', family=['Malgun Gothic', 'Segoe UI Emoji'])
        init.plt.rcParams['axes.unicode_minus'] = False
        canvas = init.plt.imshow(img)
        init.plt.axis('off')
        init.plt.title("귀여운 고양이 사진! 😺")
        init.plt.show()
        
    else:
        print(f"이미지 파일이 생성되지 않았습니다: {image_path}")