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
# from annotator.canny import CannyDetector
from annotator.openpose import OpenposeDetector
# from annotator.midas import MidasDetector
import decord
import jax
import torch

import flax.linen as nn

# apply_canny = CannyDetector()
apply_openpose = OpenposeDetector()
# apply_midas = MidasDetector()


def add_watermark(image, watermark_path, wm_rel_size=1/16, boundary=5):
    '''
    Creates a watermark on the saved inference image.
    We request that you do not remove this to properly assign credit to
    Shi-Lab's work.
    '''
    watermark = Image.open(watermark_path)
    w_0, h_0 = watermark.size
    H, W, _ = image.shape
    wmsize = int(max(H, W) * wm_rel_size)
    aspect = h_0 / w_0
    if aspect > 1.0:
        watermark = watermark.resize((wmsize, int(aspect * wmsize)), Image.LANCZOS)
    else:
        watermark = watermark.resize((int(wmsize / aspect), wmsize), Image.LANCZOS)
    w, h = watermark.size
    loc_h = H - h - boundary
    loc_w = W - w - boundary
    image = Image.fromarray(image)
    mask = watermark if watermark.mode in ('RGBA', 'LA') else None
    image.paste(watermark, (loc_w, loc_h), mask)
    return image


# def pre_process_canny(input_video, low_threshold=100, high_threshold=200):
#     detected_maps = []
#     for frame in input_video:
#         img = rearrange(frame, 'c h w -> h w c').cpu().numpy().astype(np.uint8)
#         detected_map = apply_canny(img, low_threshold, high_threshold)
#         detected_map = HWC3(detected_map)
#         detected_maps.append(detected_map[None])
#     detected_maps = np.concatenate(detected_maps)
#     control = torch.from_numpy(detected_maps.copy()).float() / 255.0
#     return rearrange(control, 'f h w c -> f c h w')


# def pre_process_depth(input_video, apply_depth_detect: bool = True):
#     detected_maps = []
#     for frame in input_video:
#         img = rearrange(frame, 'c h w -> h w c').cpu().numpy().astype(np.uint8)
#         img = HWC3(img)
#         if apply_depth_detect:
#             detected_map, _ = apply_midas(img)
#         else:
#             detected_map = img
#         detected_map = HWC3(detected_map)
#         H, W, C = img.shape
#         detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
#         detected_maps.append(detected_map[None])
#     detected_maps = np.concatenate(detected_maps)
#     control = torch.from_numpy(detected_maps.copy()).float() / 255.0
#     return rearrange(control, 'f h w c -> f c h w')


def pre_process_pose(input_video, apply_pose_detect: bool = True):
    detected_maps = []
    for frame in input_video:
        img = rearrange(frame, 'c h w -> h w c').astype(np.uint8)
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
    return rearrange(control, 'f h w c -> f c h w')


def create_video(frames, fps, rescale=False, path=None, watermark=None):
    if path is None:
        dir = "temporal"
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, 'movie.mp4')

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
        path = os.path.join(dir, 'canny_db.gif')

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

def prepare_video(video_path:str, resolution:int, device, dtype, normalize=True, start_t:float=0, end_t:float=-1, output_fps:int=-1):
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
    video = torch.Tensor(video)#.to(device).to(dtype)

    # Use max if you want the larger side to be equal to resolution (e.g. 512)
    # k = float(resolution) / min(h, w)
    k = float(resolution) / max(h, w)
    h *= k
    w *= k
    h = int(np.round(h / 64.0)) * 64
    w = int(np.round(w / 64.0)) * 64
    print(h, w)

    video = Resize((h, w), interpolation=InterpolationMode.BILINEAR, antialias=True)(video)
    if normalize:
        video = video / 127.5 - 1.0
    # video = rearrange(video, "f c h w -> f h w c").numpy() #channel first to channel last
    video = video.numpy()
    return video, output_fps


def post_process_gif(list_of_results, image_resolution):
    output_file = "/tmp/ddxk.gif"
    imageio.mimsave(output_file, list_of_results, fps=4)
    return output_file

class FlaxCrossFrameAttn(nn.Module):
    r"""
    Cross Attention 2D Mid-level block - original architecture from Unet transformers: https://arxiv.org/abs/2103.06104
    Parameters:
        in_channels (:obj:`int`):
            Input channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        attn_num_head_channels (:obj:`int`, *optional*, defaults to 1):
            Number of attention heads of each spatial transformer block
        use_memory_efficient_attention (`bool`, *optional*, defaults to `False`):
            enable memory efficient attention https://arxiv.org/abs/2112.05682
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    dropout: float = 0.0
    num_layers: int = 1
    attn_num_head_channels: int = 1
    use_linear_projection: bool = False
    use_memory_efficient_attention: bool = False
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # there is always at least one resnet
        resnets = [
            FlaxResnetBlock2D(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
        ]

        attentions = []

        for _ in range(self.num_layers):
            attn_block = FlaxTransformer2DModel(
                in_channels=self.in_channels,
                n_heads=self.attn_num_head_channels,
                d_head=self.in_channels // self.attn_num_head_channels,
                depth=1,
                use_linear_projection=self.use_linear_projection,
                use_memory_efficient_attention=self.use_memory_efficient_attention,
                dtype=self.dtype,
            )
            attentions.append(attn_block)

            res_block = FlaxResnetBlock2D(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                dropout_prob=self.dropout,
                dtype=self.dtype,
            )
            resnets.append(res_block)

        self.resnets = resnets
        self.attentions = attentions

    def __call__(self, hidden_states, temb, encoder_hidden_states, deterministic=True):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states, deterministic=deterministic)
            hidden_states = resnet(hidden_states, temb, deterministic=deterministic)

        return hidden_states
