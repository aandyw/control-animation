import imageio
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    TextToVideoZeroPipeline,
)
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import (
    CrossFrameAttnProcessor,
)
from huggingface_hub import hf_hub_download
from PIL import Image
# from safetensors.flax import load_file

assert torch.cuda.is_available() is True

device = torch.device("cuda")
print("Using device:", device.get_device_name())

if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0)/1024**3, 1), "GB")
    print("Cached:   ", round(torch.cuda.memory_reserved(0)/1024**3, 1), "GB")


model_id = "runwayml/stable-diffusion-v1-5" # base model
repo_id = "PAIR/Text2Video-Zero"
file_path = "models/ligne_claire_anime_diffusion_v1.safetensors"
filename = "__assets__/poses_skeleton_gifs/dance1_corr.mp4" # pose video

# loaded_tensors = load_file(file_path) # NOTE: use

video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)

reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
pose_images = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
).to(device)

# Set the attention processor
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

# fix latents for all frames
latents = torch.randn((1, 4, 64, 64), device=device, dtype=torch.float16).repeat(len(pose_images), 1, 1, 1)

prompt = "Darth Vader dancing in a desert"
result = pipe(prompt=[prompt] * len(pose_images), image=pose_images, latents=latents).images
imageio.mimsave("video.mp4", result, fps=4)
