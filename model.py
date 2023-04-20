from enum import Enum
import gc
import numpy as np
import jax.numpy as jnp
import tomesd
import jax
from flax.training.common_utils import shard
from flax import jax_utils
import einops

from transformers import CLIPTokenizer, CLIPFeatureExtractor, FlaxCLIPTextModel
from diffusers import FlaxDDIMScheduler, FlaxControlNetModel, FlaxUNet2DConditionModel, FlaxAutoencoderKL, FlaxStableDiffusionControlNetPipeline

import utils
import gradio_utils
import os
on_huggingspace = os.environ.get("SPACE_AUTHOR_NAME") == "PAIR"

unshard = lambda x: einops.rearrange(x, 'd b ... -> (d b) ...')

class ModelType(Enum):
    Pix2Pix_Video = 1,
    Text2Video = 2,
    ControlNetCanny = 3,
    ControlNetCannyDB = 4,
    ControlNetPose = 5,
    ControlNetDepth = 6,


class Model:
    def __init__(self, device, dtype, **kwargs):
        self.device = device
        self.dtype = dtype
        self.rng = jax.random.PRNGKey(0)
        self.pipe_dict = {
            # ModelType.Pix2Pix_Video: StableDiffusionInstructPix2PixPipeline,
            # ModelType.Text2Video: TextToVideoPipeline,
            # ModelType.ControlNetCanny: StableDiffusionControlNetPipeline,
            # ModelType.ControlNetCannyDB: StableDiffusionControlNetPipeline,
            ModelType.ControlNetPose: FlaxStableDiffusionControlNetPipeline,
            # ModelType.ControlNetDepth: StableDiffusionControlNetPipeline,
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
        self.pipe = FlaxStableDiffusionControlNetPipeline(vae=vae,
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

    def inference_chunk(self, image, frame_ids, **kwargs):
        if not hasattr(self, "pipe") or self.pipe is None:
            return
        prng_seed = jax.random.split(self.rng, jax.device_count())
        prompt = kwargs.pop('prompt')
        prompt_ids = self.pipe.prepare_text_inputs(prompt)
        negative_prompt = kwargs.pop('negative_prompt', '')
        n_prompt_ids = self.pipe.prepare_text_inputs(negative_prompt)
        latents = None
        frame_ids = jnp.array(frame_ids)
        if 'latents' in kwargs:
            latents = kwargs.pop('latents')[frame_ids]
        if self.model_type == ModelType.Text2Video:
            kwargs["frame_ids"] = frame_ids
        return self.pipe(
                        image=image[frame_ids],
                        prompt_ids=prompt_ids[frame_ids],
                        params=self.p_params,
                        prng_seed=prng_seed,
                        neg_prompt_ids=n_prompt_ids[frame_ids],
                        latents=latents[frame_ids],
                        jit=True,
                        # **kwargs
                        )

    def inference(self, image, split_to_chunks=False, chunk_size=8, **kwargs):
        if not hasattr(self, "pipe") or self.pipe is None:
            return

        if "merging_ratio" in kwargs:
            merging_ratio = kwargs.pop("merging_ratio")

            # if merging_ratio > 0:
            tomesd.apply_patch(self.pipe, ratio=merging_ratio)

        f = image.shape[0]

        assert 'prompt' in kwargs
        prompt = [kwargs.pop('prompt')] * f
        negative_prompt = [kwargs.pop('negative_prompt', '')] * f

        frames_counter = 0

        # Processing chunk-by-chunk
        if split_to_chunks:
            pass
            ## not tested
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
            prompt_ids = self.pipe.prepare_text_inputs(prompt)
            n_prompt_ids = self.pipe.prepare_text_inputs(negative_prompt)
            latents = kwargs.pop('latents')
            # rng = jax.random.split(self.rng, jax.device_count())
            prng, self.rng = jax.random.split(self.rng)
            #prng = jax.numpy.stack([prng] * jax.device_count())#same prng seed on every device
            prng = jax.random.split(prng, jax.device_count())
            return (self.pipe(image=shard(image),
                             latents=shard(latents),
                             prompt_ids=shard(prompt_ids),
                             neg_prompt_ids=shard(n_prompt_ids), 
                             params=self.p_params,
                             prng_seed=prng, jit = True,
                             **kwargs
                             ).images)[0]
        
    def process_controlnet_pose(self,
                                video_path,
                                prompt,
                                chunk_size=8,
                                #merging_ratio=0.0,
                                num_inference_steps=50,
                                controlnet_conditioning_scale=1.0,
                                guidance_scale=9.0,
                                # eta=0.0, #this doesn't exist in the flax pipeline, relates to DDIM scheduler eta
                                resolution=512,
                                save_path=None):
        print("Module Pose")
        video_path = gradio_utils.motion_to_video_path(video_path)
        if self.model_type != ModelType.ControlNetPose:
            #model_id = "tuwonga/zukki_style"
            model_id="runwayml/stable-diffusion-v1-5"
            controlnet_id = "fusing/stable-diffusion-v1-5-controlnet-openpose"
            if self.from_local:
                controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
                    controlnet_id.split("/")[-1],
                    # revision=args.controlnet_revision,
                    from_pt=True,
                    dtype=self.dtype,
                )
            else:
                controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
                    controlnet_id,
                    # revision=args.controlnet_revision,
                    from_pt=True,
                    dtype=self.dtype,
                )
            tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
            )
            scheduler, scheduler_state = FlaxDDIMScheduler.from_pretrained(
            "runwayml/stable-diffusion-v1-5", subfolder ="scheduler"
            )
            self.set_model(ModelType.ControlNetPose,
                            model_id=model_id,
                            tokenizer=tokenizer,
                            controlnet=controlnet,
                            controlnet_params=controlnet_params,
                            scheduler=scheduler,
                            scheduler_state=scheduler_state)

        video_path = gradio_utils.motion_to_video_path(
            video_path) if 'Motion' in video_path else video_path

        added_prompt = 'best quality, extremely detailed, HD, ultra-realistic, 8K, HQ, masterpiece, trending on artstation, art, smooth'
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic'

        video, fps = utils.prepare_video(
            video_path, resolution, self.device, self.dtype, False, output_fps=4)
        control = utils.pre_process_pose(
            video, apply_pose_detect=False)
        f, _, h, w = video.shape

        control = jnp.expand_dims(control, 0).repeat(jax.device_count(), 0)

        self.rng, latents_rng = jax.random.split(self.rng)
        latents = jax.random.normal(latents_rng, (1, 1, 4, h//8, w//8))
        latents = latents.repeat(f, 1) #latents.repeat(f, 1, 1, 1)
        latents = latents.repeat(jax.device_count(), 0)

        result = self.inference(image=control,
                                prompt=prompt + ', ' + added_prompt,
                                # height=h,
                                # width=w,
                                negative_prompt=negative_prompts,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                controlnet_conditioning_scale=controlnet_conditioning_scale,
                                # eta=eta,
                                latents=latents,
                                # output_type='numpy',
                                split_to_chunks=False,
                                chunk_size=chunk_size,
                                # merging_ratio=merging_ratio,
                                )
        return utils.create_gif(result, fps, path=save_path, watermark=None)
