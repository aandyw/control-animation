import torch
import imageio
from diffusers import (
    TextToVideoZeroPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    TextToVideoZeroPipeline,
)
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import (
    CrossFrameAttnProcessor,
)
from huggingface_hub import hf_hub_download
from PIL import Image


from text_to_animation.model import ControlAnimationModel

model = ControlAnimationModel(device="cuda", dtype=torch.float16)

controlnet_video, initial_frames = model.generate_initial_frames(
    prompt="astronaut walking through the forest",
    video_path="Motion 1",
    n_prompt="",
    seed=0,
)

print(controlnet_video, initial_frames)
