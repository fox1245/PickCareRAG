import init
init.os.environ['ATTN_BACKEND'] = 'flash-attn'
init.os.environ['SPCONV_ALGO'] = 'native' 
import imageio 
from PIL import Image
from TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from TRELLIS.trellis.utils import render_utils, postprocessing_utils

#hugginfface 모델 허브에서 모델 끌어오기
pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
pipeline.cuda()



#이미지 로드
def create_3d_from_image(file_path, seed = 1, steps = 12, cfg_strength = 7.5, output_file = "sample_mesh.mp4", fps = 30):
    image = Image.open(file_path)
    #파이프라인 구동
    outputs = pipeline.run(
        image,
        seed = seed,
        sparse_structure_sampler_params={
        "steps": steps,
        "cfg_strength": cfg_strength,
        },
    )
    #이미지 렌더링
    video = render_utils.render_video(outputs['mesh'][0])["normal"]
    imageio.mimsave(output_file, video, fps = fps)
    
    
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,  
    )
    output_file_glb = output_file.replace(".mp4", ".glb")
    glb.export(output_file_glb)
    
     

