from enum import Enum
import gc
import numpy as np
import tomesd
import torch

from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UNet2DConditionModel,
)
from diffusers.schedulers import EulerAncestralDiscreteScheduler, DDIMScheduler
from text_to_animation.pipelines.text_to_video_pipeline import TextToVideoPipeline

import utils.utils as utils
import utils.gradio_utils as gradio_utils
import os

on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"

from einops import rearrange


class ModelType(Enum):
    Pix2Pix_Video = (1,)
    Text2Video = (2,)
    ControlNetCanny = (3,)
    ControlNetCannyDB = (4,)
    ControlNetPose = (5,)
    ControlNetDepth = (6,)


class ControlAnimationModel:
    def __init__(self, device, dtype, **kwargs):
        self.device = device
        self.dtype = dtype
        self.generator = torch.Generator(device=device)
        self.controlnet_attn_proc = utils.CrossFrameAttnProcessor(unet_chunk_size=2)
        self.pix2pix_attn_proc = utils.CrossFrameAttnProcessor(unet_chunk_size=3)
        self.text2video_attn_proc = utils.CrossFrameAttnProcessor(unet_chunk_size=2)

        self.pipe = None
        self.model_type = None

        self.states = {}
        self.model_name = ""

    def set_model(self, model_type: ModelType, model_id: str, **kwargs):
        if hasattr(self, "pipe") and self.pipe is not None:
            del self.pipe
        torch.cuda.empty_cache()
        gc.collect()
        safety_checker = kwargs.pop("safety_checker", None)
        self.pipe = (
            self.pipe_dict[model_type]
            .from_pretrained(model_id, safety_checker=safety_checker, **kwargs)
            .to(self.device)
            .to(self.dtype)
        )
        self.model_type = model_type
        self.model_name = model_id

    def inference_chunk(self, frame_ids, **kwargs):
        if not hasattr(self, "pipe") or self.pipe is None:
            return

        prompt = np.array(kwargs.pop("prompt"))
        negative_prompt = np.array(kwargs.pop("negative_prompt", ""))
        latents = None
        if "latents" in kwargs:
            latents = kwargs.pop("latents")[frame_ids]
        if "image" in kwargs:
            kwargs["image"] = kwargs["image"][frame_ids]
        if "video_length" in kwargs:
            kwargs["video_length"] = len(frame_ids)
        if self.model_type == ModelType.Text2Video:
            kwargs["frame_ids"] = frame_ids
        return self.pipe(
            prompt=prompt[frame_ids].tolist(),
            negative_prompt=negative_prompt[frame_ids].tolist(),
            latents=latents,
            generator=self.generator,
            **kwargs,
        )

    def inference(self, split_to_chunks=False, chunk_size=2, **kwargs):
        if not hasattr(self, "pipe") or self.pipe is None:
            return

        if "merging_ratio" in kwargs:
            merging_ratio = kwargs.pop("merging_ratio")

            # if merging_ratio > 0:
            tomesd.apply_patch(self.pipe, ratio=merging_ratio)
        seed = kwargs.pop("seed", 0)
        if seed < 0:
            seed = self.generator.seed()
        kwargs.pop("generator", "")

        if "image" in kwargs:
            f = kwargs["image"].shape[0]
        else:
            f = kwargs["video_length"]

        assert "prompt" in kwargs
        prompt = [kwargs.pop("prompt")] * f
        negative_prompt = [kwargs.pop("negative_prompt", "")] * f

        frames_counter = 0

        # Processing chunk-by-chunk
        if split_to_chunks:
            chunk_ids = np.arange(0, f, chunk_size - 1)
            result = []
            for i in range(len(chunk_ids)):
                ch_start = chunk_ids[i]
                ch_end = f if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
                frame_ids = [0] + list(range(ch_start, ch_end))
                self.generator.manual_seed(seed)
                print(f"Processing chunk {i + 1} / {len(chunk_ids)}")
                result.append(
                    self.inference_chunk(
                        frame_ids=frame_ids,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        **kwargs,
                    ).images[1:]
                )
                frames_counter += len(chunk_ids) - 1
                if on_huggingspace and frames_counter >= 80:
                    break
            result = np.concatenate(result)
            return result
        else:
            self.generator.manual_seed(seed)
            return self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                generator=self.generator,
                **kwargs,
            ).images

    def build_prompt(self, prompt, base_prompt):
        new_prompt = prompt.rstrip()
        if len(new_prompt) > 0 and (new_prompt[-1] == "," or new_prompt[-1] == "."):
            new_prompt = new_prompt.rstrip()[:-1]
        new_prompt = new_prompt.rstrip()
        new_prompt = new_prompt + ", " + base_prompt

        return new_prompt

    def generate_initial_frames(
        self,
        video_path,
        prompt,
        model_name="dreamlike-art/dreamlike-photoreal-2.0",
        motion_field_strength_x=12,
        motion_field_strength_y=12,
        t0=44,
        t1=47,
        n_prompt="",
        chunk_size=2,
        video_length=8,
        watermark="Picsart AI Research",
        merging_ratio=0.0,
        seed=0,
        resolution=512,
        fps=2,
        use_cf_attn=True,
        use_motion_field=True,
        smooth_bg=False,
        smooth_bg_strength=0.4,
        path=None,
    ):
        print("Generating Initial Frames...")
        unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
        self.set_model(ModelType.Text2Video, model_id=model_name, unet=unet)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        if use_cf_attn:
            self.pipe.unet.set_attn_processor(processor=self.text2video_attn_proc)

        self.generator.manual_seed(seed)

        added_prompt = "high quality, HD, 8K, trending on artstation, high focus, dramatic lighting"
        negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"

        prompt = self.build_prompt(prompt, added_prompt)
        negative_prompt = self.build_prompt(prompt, negative_prompt)

        video, fps = utils.prepare_video(
            video_path, resolution, self.device, self.dtype, False, output_fps=4
        )
        control = (
            utils.process_first_frame(video, apply_pose_detect=False)
            .to(self.device)
            .to(self.dtype)
        )
        f, _, h, w = video.shape

        result = self.inference(
            image=control,
            prompt=prompt,
            video_length=video_length,
            height=h,
            width=w,
            num_inference_steps=50,
            guidance_scale=7.5,
            guidance_stop_step=1.0,
            t0=t0,
            t1=t1,
            motion_field_strength_x=motion_field_strength_x,
            motion_field_strength_y=motion_field_strength_y,
            use_motion_field=use_motion_field,
            smooth_bg=smooth_bg,
            smooth_bg_strength=smooth_bg_strength,
            seed=seed,
            output_type="numpy",
            negative_prompt=negative_prompt,
            merging_ratio=merging_ratio,
            split_to_chunks=True,
            chunk_size=chunk_size,
        )
        return utils.create_video(
            result, fps, path=path, watermark=gradio_utils.logo_name_to_path(watermark)
        )
