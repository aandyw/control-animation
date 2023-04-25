import torch
from enum import Enum
import gc
import numpy as np
import jax.numpy as jnp
import tomesd
import jax
# from flax.training.common_utils import shard
from flax import jax_utils
import einops

from transformers import CLIPTokenizer, CLIPFeatureExtractor, FlaxCLIPTextModel
from diffusers import FlaxDDIMScheduler, FlaxControlNetModel, FlaxUNet2DConditionModel, FlaxAutoencoderKL, FlaxStableDiffusionControlNetPipeline
from flax_text_to_video_pipeline import FlaxTextToVideoPipeline

import utils.utils
import utils.gradio_utils as gradio_utils
import os
on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"

unshard = lambda x: einops.rearrange(x, 'd b ... -> (d b) ...')

class ModelType(Enum):
    Text2Video = 1,
    ControlNetPose = 2,


def replicate_devices(array):
    return jnp.expand_dims(array, 0).repeat(jax.device_count(), 0)

class ControlAnimationModel:
    def __init__(self, device, dtype, **kwargs):
        self.device = device
        self.dtype = dtype
        self.rng = jax.random.PRNGKey(0)
        self.pipe_dict = {
            ModelType.Text2Video: FlaxTextToVideoPipeline, # TODO: Replace with our TextToVideo JAX Pipeline
            ModelType.ControlNetPose: FlaxStableDiffusionControlNetPipeline,
        }
        self.pipe = None
        self.model_type = None

        self.states = {}
        self.model_name = ""

        self.from_local = True #if the attn model is available in local (after adaptation by adapt_attn.py)

    def set_model(self, model_type: ModelType, model_id: str, controlnet, controlnet_params, tokenizer, scheduler, scheduler_state, **kwargs):
        if hasattr(self, "pipe") and self.pipe is not None:
            del self.pipe
            self.pipe = None
        gc.collect()
        scheduler, scheduler_state = FlaxDDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            from_pt =True
        )
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        feature_extractor = CLIPFeatureExtractor.from_pretrained(model_id, subfolder="feature_extractor")
        if self.from_local:
            unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(f'./{model_id.split("/")[-1]}', subfolder="unet", from_pt=True, dtype=self.dtype)
        else:
            unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(model_id, subfolder="unet", from_pt=True, dtype=self.dtype)
        vae, vae_params = FlaxAutoencoderKL.from_pretrained(model_id, subfolder="vae", from_pt=True, dtype=self.dtype)
        text_encoder = FlaxCLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", from_pt=True, dtype=self.dtype)
        self.pipe = FlaxTextToVideoPipeline(vae=vae,
                                                        text_encoder=text_encoder,
                                                        tokenizer=tokenizer,
                                                        unet=unet,
                                                        controlnet=controlnet,
                                                        scheduler=scheduler,
                                                        safety_checker=None,
                                                        feature_extractor=feature_extractor)
        self.params = {'unet': unet_params,
                "vae": vae_params,
                "scheduler": scheduler_state,
                "controlnet": controlnet_params,
                "text_encoder": text_encoder.params}
        self.p_params = jax_utils.replicate(self.params)

        self.model_type = model_type
        self.model_name = model_id

    # def inference_chunk(self, image, frame_ids, prompt, negative_prompt, **kwargs):

    #     prompt_ids = self.pipe.prepare_text_inputs(prompt)
    #     n_prompt_ids = self.pipe.prepare_text_inputs(negative_prompt)
    #     latents = kwargs.pop('latents')
    #     # rng = jax.random.split(self.rng, jax.device_count())
    #     prng, self.rng = jax.random.split(self.rng)
    #     #prng = jax.numpy.stack([prng] * jax.device_count())#same prng seed on every device
    #     prng_seed = jax.random.split(prng, jax.device_count())
    #     image = replicate_devices(image[frame_ids])
    #     latents = replicate_devices(latents)
    #     prompt_ids = replicate_devices(prompt_ids)
    #     n_prompt_ids = replicate_devices(n_prompt_ids)
    #     return (self.pipe(image=image,
    #                         latents=latents,
    #                         prompt_ids=prompt_ids,
    #                         neg_prompt_ids=n_prompt_ids, 
    #                         params=self.p_params,
    #                         prng_seed=prng_seed, jit = True,
    #                         ).images)[0]

    def inference(self, image, split_to_chunks=False, chunk_size=8, **kwargs):
        if not hasattr(self, "pipe") or self.pipe is None:
            return

        if "merging_ratio" in kwargs:
            merging_ratio = kwargs.pop("merging_ratio")

            # if merging_ratio > 0:
            tomesd.apply_patch(self.pipe, ratio=merging_ratio)

        # f = image.shape[0]

        assert 'prompt' in kwargs
        prompt = [kwargs.pop('prompt')] 
        negative_prompt = [kwargs.pop('negative_prompt', '')]

        frames_counter = 0

        # Processing chunk-by-chunk
        if split_to_chunks:
            pass
            # # not tested
            # f = image.shape[0]
            # chunk_ids = np.arange(0, f, chunk_size - 1)
            # result = []
            # for i in range(len(chunk_ids)):
            #     ch_start = chunk_ids[i]
            #     ch_end = f if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
            #     frame_ids = [0] + list(range(ch_start, ch_end))
            #     print(f'Processing chunk {i + 1} / {len(chunk_ids)}')
            #     result.append(self.inference_chunk(image=image,
            #                                        frame_ids=frame_ids,
            #                                        prompt=prompt,
            #                                        negative_prompt=negative_prompt,
            #                                        **kwargs).images[1:])
            #     frames_counter += len(chunk_ids)-1
            #     if on_huggingspace and frames_counter >= 80:
            #         break
            # result = np.concatenate(result)
            # return result
        else:
            if 'jit' in kwargs and kwargs.pop('jit'):
                prompt_ids = self.pipe.prepare_text_inputs(prompt)
                n_prompt_ids = self.pipe.prepare_text_inputs(negative_prompt)
                latents = kwargs.pop('latents')
                prng, self.rng = jax.random.split(self.rng)
                prng_seed = jax.random.split(prng, jax.device_count())
                image = replicate_devices(image)
                latents = replicate_devices(latents)
                prompt_ids = replicate_devices(prompt_ids)
                n_prompt_ids = replicate_devices(n_prompt_ids)
                return (self.pipe(image=image,
                                latents=latents,
                                prompt_ids=prompt_ids,
                                neg_prompt_ids=n_prompt_ids, 
                                params=self.p_params,
                                prng_seed=prng_seed, jit = True,
                                ).images)[0]
            else:
                prompt_ids = self.pipe.prepare_text_inputs(prompt)
                n_prompt_ids = self.pipe.prepare_text_inputs(negative_prompt)
                latents = kwargs.pop('latents')
                prng_seed, self.rng = jax.random.split(self.rng)
                return self.pipe(image=image,
                                latents=latents,
                                prompt_ids=prompt_ids,
                                neg_prompt_ids=n_prompt_ids, 
                                params=self.p_params,
                                prng_seed=prng_seed, jit = False,
                                ).images

    def process_controlnet_pose(self,
                                video_path,
                                prompt,
                                chunk_size=8,
                                watermark='Picsart AI Research',
                                merging_ratio=0.0,
                                num_inference_steps=20,
                                controlnet_conditioning_scale=1.0,
                                guidance_scale=9.0,
                                seed=42,
                                eta=0.0,
                                resolution=512,
                                use_cf_attn=True,
                                save_path=None):
        print("Module Pose")
        video_path = gradio_utils.motion_to_video_path(video_path)
        if self.model_type != ModelType.ControlNetPose:
            controlnet = FlaxControlNetModel.from_pretrained(
                "fusing/stable-diffusion-v1-5-controlnet-openpose")
            self.set_model(ModelType.ControlNetPose,
                           model_id="runwayml/stable-diffusion-v1-5", controlnet=controlnet)
            self.pipe.scheduler = FlaxDDIMScheduler.from_config(
                self.pipe.scheduler.config)
            if use_cf_attn:
                self.pipe.unet.set_attn_processor(
                    processor=self.controlnet_attn_proc)
                self.pipe.controlnet.set_attn_processor(
                    processor=self.controlnet_attn_proc)

        video_path = gradio_utils.motion_to_video_path(
            video_path) if 'Motion' in video_path else video_path

        added_prompt = 'best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth'
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic'

        video, fps = utils.prepare_video(
            video_path, resolution, self.device, self.dtype, False, output_fps=4)
        control = utils.pre_process_pose(
            video, apply_pose_detect=False).to(self.device).to(self.dtype)
        f, _, h, w = video.shape
        self.generator.manual_seed(seed)
        latents = torch.randn((1, 4, h//8, w//8), dtype=self.dtype,
                              device=self.device, generator=self.generator)
        latents = latents.repeat(f, 1, 1, 1)
        result = self.inference(image=control,
                                prompt=prompt + ', ' + added_prompt,
                                height=h,
                                width=w,
                                negative_prompt=negative_prompts,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                controlnet_conditioning_scale=controlnet_conditioning_scale,
                                eta=eta,
                                latents=latents,
                                seed=seed,
                                output_type='numpy',
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                merging_ratio=merging_ratio,
                                )
        return utils.create_gif(result, fps, path=save_path, watermark=gradio_utils.logo_name_to_path(watermark))

    def process_text2video(self,
                           prompt,
                           model_name="dreamlike-art/dreamlike-photoreal-2.0",
                           motion_field_strength_x=12,
                           motion_field_strength_y=12,
                           t0=44,
                           t1=47,
                           n_prompt="",
                           chunk_size=8,
                           video_length=8,
                           watermark='Picsart AI Research',
                           merging_ratio=0.0,
                           seed=0,
                           resolution=512,
                           fps=2,
                           use_cf_attn=True,
                           use_motion_field=True,
                           smooth_bg=False,
                           smooth_bg_strength=0.4,
                           path=None):
        print("Module Text2Video")
        if self.model_type != ModelType.Text2Video or model_name != self.model_name:
            print("Model update")
            unet = FlaxUNet2DConditionModel.from_pretrained(
                model_name, subfolder="unet")
            self.set_model(ModelType.Text2Video,
                           model_id=model_name, unet=unet)
            self.pipe.scheduler = FlaxDDIMScheduler.from_config(
                self.pipe.scheduler.config)
            if use_cf_attn:
                self.pipe.unet.set_attn_processor(
                    processor=self.text2video_attn_proc)
        self.generator.manual_seed(seed)

        added_prompt = "high quality, HD, 8K, trending on artstation, high focus, dramatic lighting"
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic'

        prompt = prompt.rstrip()
        if len(prompt) > 0 and (prompt[-1] == "," or prompt[-1] == "."):
            prompt = prompt.rstrip()[:-1]
        prompt = prompt.rstrip()
        prompt = prompt + ", "+added_prompt
        if len(n_prompt) > 0:
            negative_prompt = n_prompt
        else:
            negative_prompt = None

        result = self.inference(prompt=prompt,
                                video_length=video_length,
                                height=resolution,
                                width=resolution,
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
                                output_type='numpy',
                                negative_prompt=negative_prompt,
                                merging_ratio=merging_ratio,
                                split_to_chunks=True,
                                chunk_size=chunk_size,
                                )
        return utils.create_video(result, fps, path=path, watermark=gradio_utils.logo_name_to_path(watermark))
    
