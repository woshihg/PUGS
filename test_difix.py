from Difix3D.src.pipeline_difix import DifixPipeline
from diffusers.utils import load_image
import time

pipe = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
pipe.to("cuda")

# 记录开始时间
start_time = time.time()
input_image = load_image("/home/woshihg/PycharmProjects/PUGS/output/model_2025-12-02_10-47-36/difix_output_iter_00001.png")
print(f"Input image type: {type(input_image)}, size: {input_image.size}")
ref_image = load_image("/home/woshihg/PycharmProjects/Difix3D/assets/DSC_6487.jpg")
# 将ref_image调整为与input_image相同的大小
ref_image = ref_image.resize(input_image.size)
prompt = "remove degradation"

output_image = pipe(prompt,
                    image=input_image,
                    # ref_image=ref_image,
                    num_inference_steps=1,
                    timesteps=[199],
                    guidance_scale=0.0).images[0]
output_image.save("/home/woshihg/PycharmProjects/Difix3D/assets/difix_output_iter_00001.png")
# 记录结束时间
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")