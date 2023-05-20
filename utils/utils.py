import os

import PIL.Image
import numpy as np
import torchvision
from torchvision.transforms import Resize, InterpolationMode
import imageio
from einops import rearrange
import cv2
from PIL import Image
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
import decord
import jax
import torch
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    download_from_original_stable_diffusion_ckpt,
)
from huggingface_hub import hf_hub_download

import flax.linen as nn

apply_openpose = OpenposeDetector()

# TODO

# def load_safetensors_model(model_link):
#     ckpt_path = hf_hub_download(
#         repo_id=model_link, filename="ligne_claire_anime_diffusion_v1.safetensors"
#     )

#     print(f"Checkpoint path: {ckpt_path}")

#     # !wget https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml
#     pipe = download_from_original_stable_diffusion_ckpt(
#         checkpoint_path=ckpt_path,
#         original_config_file="v1-inference.yaml",
#         from_safetensors=True,
#     )
#     pipe.save_pretrained("./models/ligne_claire", safe_serialization=True)

#     return pipe


def process_first_frame(input_video, apply_pose_detect: bool = True):
    frame = input_video[0]  # get first frame
    img = rearrange(frame, "c h w -> h w c").astype(np.uint8)
    img = HWC3(img)
    if apply_pose_detect:
        detected_map, _ = apply_openpose(img)
    else:
        detected_map = img
    detected_map = HWC3(detected_map)
    H, W, C = img.shape
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
    control = (detected_map[None].copy()) / 255.0
    return rearrange(control, "f h w c -> f c h w")


def pre_process_pose(input_video, apply_pose_detect: bool = True):
    detected_maps = []
    for frame in input_video:
        img = rearrange(frame, "c h w -> h w c").astype(np.uint8)
        img = HWC3(img)
        if apply_pose_detect:
            detected_map, _ = apply_openpose(img)
        else:
            detected_map = img
        detected_map = HWC3(detected_map)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps)
    control = (detected_maps.copy()) / 255.0
    return rearrange(control, "f h w c -> f c h w")


def create_video(frames, fps, rescale=False, path=None, watermark=None):
    if path is None:
        dir = "temporal"
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, "movie.mp4")

    outputs = []
    for i, x in enumerate(frames):
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)

        if watermark is not None:
            x = add_watermark(x, watermark)
        outputs.append(x)
        # imageio.imsave(os.path.join(dir, os.path.splitext(name)[0] + f'_{i}.jpg'), x)

    imageio.mimsave(path, outputs, fps=fps)
    return path


def create_gif(frames, fps, rescale=False, path=None, watermark=None):
    if path is None:
        dir = "temporal"
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, "canny_db.gif")

    outputs = []
    for i, x in enumerate(frames):
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        if watermark is not None:
            x = add_watermark(x, watermark)
        outputs.append(x)
        # imageio.imsave(os.path.join(dir, os.path.splitext(name)[0] + f'_{i}.jpg'), x)

    imageio.mimsave(path, outputs, fps=fps)
    return path


def prepare_video(
    video_path: str,
    resolution: int,
    device,
    dtype,
    normalize=True,
    start_t: float = 0,
    end_t: float = -1,
    output_fps: int = -1,
):
    vr = decord.VideoReader(video_path)
    initial_fps = vr.get_avg_fps()
    if output_fps == -1:
        output_fps = int(initial_fps)
    if end_t == -1:
        end_t = len(vr) / initial_fps
    else:
        end_t = min(len(vr) / initial_fps, end_t)
    assert 0 <= start_t < end_t
    assert output_fps > 0
    start_f_ind = int(start_t * initial_fps)
    end_f_ind = int(end_t * initial_fps)
    num_f = int((end_t - start_t) * output_fps)
    sample_idx = np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)
    video = vr.get_batch(sample_idx)
    video = video.asnumpy()
    _, h, w, _ = video.shape
    video = rearrange(video, "f h w c -> f c h w")
    video = torch.Tensor(video)  # .to(device).to(dtype)

    # Use max if you want the larger side to be equal to resolution (e.g. 512)
    # k = float(resolution) / min(h, w)
    k = float(resolution) / max(h, w)
    h *= k
    w *= k
    h = int(np.round(h / 64.0)) * 64
    w = int(np.round(w / 64.0)) * 64

    video = Resize((h, w), interpolation=InterpolationMode.BILINEAR, antialias=True)(
        video
    )
    if normalize:
        video = video / 127.5 - 1.0
    # video = rearrange(video, "f c h w -> f h w c").numpy() #channel first to channel last
    video = video.numpy()
    return video, output_fps


def post_process_gif(list_of_results, image_resolution):
    output_file = "/tmp/ddxk.gif"
    imageio.mimsave(output_file, list_of_results, fps=4)
    return output_file
