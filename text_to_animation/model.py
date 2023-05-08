import torch
from enum import Enum
import gc
import numpy as np
import jax.numpy as jnp
import jax

from PIL import Image
from typing import List

from flax.training.common_utils import shard
from flax.jax_utils import replicate
from flax import jax_utils
import einops

from transformers import CLIPTokenizer, CLIPFeatureExtractor, FlaxCLIPTextModel
from diffusers import (
    FlaxDDIMScheduler,
    FlaxAutoencoderKL,
    FlaxStableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    FlaxUNet2DConditionModel as VanillaFlaxUNet2DConditionModel,
)
from text_to_animation.models.unet_2d_condition_flax import (
    FlaxUNet2DConditionModel
)
from diffusers import FlaxControlNetModel

from text_to_animation.pipelines.text_to_video_pipeline_flax import (
    FlaxTextToVideoPipeline,
)

import utils.utils as utils
import utils.gradio_utils as gradio_utils
import os

on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"

unshard = lambda x: einops.rearrange(x, "d b ... -> (d b) ...")


class ModelType(Enum):
    Text2Video = 1
    ControlNetPose = 2
    StableDiffusion = 3


def replicate_devices(array):
    return jnp.expand_dims(array, 0).repeat(jax.device_count(), 0)


class ControlAnimationModel:
    def __init__(self, dtype, **kwargs):
        self.dtype = dtype
        self.rng = jax.random.PRNGKey(0)
        self.pipe = None
        self.model_type = None

        self.states = {}
        self.model_name = ""

    def set_model(
        self,
        model_id: str,
        **kwargs,
    ):
        if hasattr(self, "pipe") and self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()

        controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
            "fusing/stable-diffusion-v1-5-controlnet-openpose",
            from_pt=True,
            dtype=jnp.float16,
        )

        scheduler, scheduler_state = FlaxDDIMScheduler.from_pretrained(
            model_id, subfolder="scheduler", from_pt=True
        )
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        feature_extractor = CLIPFeatureExtractor.from_pretrained(
            model_id, subfolder="feature_extractor"
        )
        unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", from_pt=True, dtype=self.dtype
        )
        unet_vanilla = VanillaFlaxUNet2DConditionModel.from_config(
            model_id, subfolder="unet", from_pt=True, dtype=self.dtype
        )
        vae, vae_params = FlaxAutoencoderKL.from_pretrained(
            model_id, subfolder="vae", from_pt=True, dtype=self.dtype
        )
        text_encoder = FlaxCLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", from_pt=True, dtype=self.dtype
        )
        self.pipe = FlaxTextToVideoPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            unet_vanilla=unet_vanilla,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=feature_extractor,
        )
        self.params = {
            "unet": unet_params,
            "vae": vae_params,
            "scheduler": scheduler_state,
            "controlnet": controlnet_params,
            "text_encoder": text_encoder.params,
        }
        self.p_params = jax_utils.replicate(self.params)
        self.model_name = model_id

    def generate_initial_frames(
        self,
        prompt: str,
        video_path: str,
        n_prompt: str = "",
        num_imgs: int = 4,
        resolution: int = 512,
        model_id: str = "runwayml/stable-diffusion-v1-5",
    ) -> List[Image.Image]:
        self.set_model(model_id=model_id)

        video_path = gradio_utils.motion_to_video_path(video_path)

        added_prompt = "high quality, best quality, HD, clay stop-motion, claymation, HQ, masterpiece, art, smooth"
        prompts = added_prompt + ", " + prompt

        added_n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly"
        negative_prompts = added_n_prompt + ", " + n_prompt

        video, fps = utils.prepare_video(
            video_path, resolution, None, self.dtype, False, output_fps=4
        )
        control = utils.pre_process_pose(video, apply_pose_detect=False)

        seeds = [seed for seed in jax.random.randint(self.rng, [num_imgs], 0, 65536)]
        prngs = [jax.random.PRNGKey(seed) for seed in seeds]
        print(seeds)
        images = self.pipe.generate_starting_frames(
            params=self.p_params,
            prngs=prngs,
            controlnet_image=control,
            prompt=prompts,
            neg_prompt=negative_prompts,
        )

        images = [np.array(images[i]) for i in range(images.shape[0])]

        return images

    def generate_video_from_frame(self, controlnet_video, prompt, seed, neg_prompt=""):
        # generate a video using the seed provided
        prng_seed = jax.random.PRNGKey(seed)
        len_vid = controlnet_video.shape[0]
        # print(f"Generating video from prompt {'<aardman> style '+ prompt}, with {controlnet_video.shape[0]} frames and prng seed {seed}")
        added_prompt = "high quality, best quality, HD, clay stop-motion, claymation, HQ, masterpiece, art, smooth"
        prompts = added_prompt + ", " + prompt

        added_n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly"
        negative_prompts = added_n_prompt + ", " + neg_prompt
        
        # prompt_ids = self.pipe.prepare_text_inputs(["aardman style "+ prompt]*len_vid)
        # n_prompt_ids = self.pipe.prepare_text_inputs([neg_prompt]*len_vid)
        
        prompt_ids = self.pipe.prepare_text_inputs([prompts]*len_vid)
        n_prompt_ids = self.pipe.prepare_text_inputs([negative_prompts]*len_vid)
        prng = replicate_devices(prng_seed) #jax.random.split(prng, jax.device_count())
        image = replicate_devices(controlnet_video)
        prompt_ids = replicate_devices(prompt_ids)
        n_prompt_ids = replicate_devices(n_prompt_ids)
        motion_field_strength_x = replicate_devices(jnp.array(3))
        motion_field_strength_y = replicate_devices(jnp.array(4))
        smooth_bg_strength = replicate_devices(jnp.array(0.8))
        vid = (self.pipe(image=image,
                        prompt_ids=prompt_ids,
                        neg_prompt_ids=n_prompt_ids, 
                        params=self.p_params,
                        prng_seed=prng,
                        jit = True,
                        smooth_bg_strength=smooth_bg_strength,
                        motion_field_strength_x=motion_field_strength_x,
                        motion_field_strength_y=motion_field_strength_y,
                        ).images)[0]
        return utils.create_gif(np.array(vid), 4, path=None, watermark=None)


    def generate_animation(
        self,
        prompt, #: str,
        initial_frame_index, #: int,
        input_video_path, #: str,
        model_link = None,#: str = "dreamlike-art/dreamlike-photoreal-2.0",
        motion_field_strength_x = 12,#: int = 12,
        motion_field_strength_y= 12,#: int = 12,
        t0= 44,#: int = 44,
        t1= 47,#: int = 47,
        n_prompt= "",#: str = "",
        chunk_size= 8, #: int = 8,
        video_length = 8, #: int = 8,
        merging_ratio = 0., #: float = 0.0,
        seed= 0,#: int = 0,
        resolution=512,#: int = 512,
        fps=2,#: int = 2,
        use_cf_attn=True,#: bool = True,
        use_motion_field=True,#: bool = True,
        smooth_bg=False,#: bool = False,
        smooth_bg_strength=0.4,#: float = 0.4,
        path=None,#: str = None,
    ):
        video_path = gradio_utils.motion_to_video_path(video_path)

        # added_prompt = 'best quality, HD, clay stop-motion, claymation, HQ, masterpiece, art, smooth'
        # added_prompt = 'high quality, anatomically correct, clay stop-motion, aardman, claymation, smooth'
        added_n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly"
        negative_prompts = added_n_prompt + ", " + n_prompt

        video, fps = utils.prepare_video(
            input_video_path, resolution, None, self.dtype, False, output_fps=4
        )
        control = utils.pre_process_pose(video, apply_pose_detect=False)
        len_vid, _, h, w = video.shape
        prng_seed = jax.random.PRNGKey(seed)
        prompts = prompt
        prompt_ids = self.pipe.prepare_text_inputs([prompts]*len_vid)
        n_prompt_ids = self.pipe.prepare_text_inputs([negative_prompts]*len_vid)
        prng = replicate_devices(prng_seed) #jax.random.split(prng, jax.device_count())
        image = replicate_devices(control)
        prompt_ids = replicate_devices(prompt_ids)
        n_prompt_ids = replicate_devices(n_prompt_ids)
        motion_field_strength_x = replicate_devices(jnp.array(motion_field_strength_x))
        motion_field_strength_y = replicate_devices(jnp.array(motion_field_strength_y))
        smooth_bg_strength = replicate_devices(jnp.array(smooth_bg_strength))
        vid = (self.pipe(image=image,
                        prompt_ids=prompt_ids,
                        neg_prompt_ids=n_prompt_ids, 
                        params=self.p_params,
                        prng_seed=prng,
                        jit = True,
                        smooth_bg_strength=smooth_bg_strength,
                        motion_field_strength_x=motion_field_strength_x,
                        motion_field_strength_y=motion_field_strength_y,
                        ).images)[0]
        return utils.create_gif(np.array(vid), 4, path=None, watermark=None)
