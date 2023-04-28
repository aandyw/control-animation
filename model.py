from enum import Enum
import gc
import numpy as np
import jax.numpy as jnp
# import tomesd
import jax
# from flax.training.common_utils import shard
from flax import jax_utils
import einops

from transformers import CLIPTokenizer, CLIPFeatureExtractor, FlaxCLIPTextModel
from diffusers import FlaxDDIMScheduler, FlaxAutoencoderKL #FlaxUNet2DConditionModel, FlaxStableDiffusionControlNetPipeline

from custom_flaxunet2D.unet_2d_condition_flax import FlaxUNet2DConditionModel
from custom_flaxunet2D.controlnet_flax import FlaxControlNetModel
from flax_text_to_video_pipeline import FlaxTextToVideoPipeline

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


def replicate_devices(array):
    return jnp.expand_dims(array, 0).repeat(jax.device_count(), 0)

class Model:
    def __init__(self, device, dtype, **kwargs):
        self.device = device
        self.dtype = dtype
        self.rng = jax.random.PRNGKey(0)

        self.pipe = None
        self.model_type = None

        self.states = {}
        self.model_name = ""

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
                # latents = kwargs.pop('latents')
                prng, self.rng = jax.random.split(self.rng)
                prng_seed = jax.random.split(prng, jax.device_count())
                image = replicate_devices(image)
                # latents = replicate_devices(latents)
                prompt_ids = replicate_devices(prompt_ids)
                n_prompt_ids = replicate_devices(n_prompt_ids)
                motion_field_strength_x = replicate_devices(kwargs.pop("motion_field_strength_x"))
                motion_field_strength_y = replicate_devices(kwargs.pop("motion_field_strength_y"))
                return (self.pipe(image=image,
                                # latents=latents,
                                prompt_ids=prompt_ids,
                                neg_prompt_ids=n_prompt_ids, 
                                params=self.p_params,
                                prng_seed=prng_seed, jit = True,
                                motion_field_strength_x=motion_field_strength_x,
                                motion_field_strength_y=motion_field_strength_y,
                                **kwargs
                                ).images)[0]
            else:
                print("no jit")
                prompt_ids = self.pipe.prepare_text_inputs(prompt)
                n_prompt_ids = self.pipe.prepare_text_inputs(negative_prompt)
                # latents = kwargs.pop('latents')
                prng_seed, self.rng = jax.random.split(self.rng)
                return self.pipe(image=image,
                                # latents=latents,
                                prompt_ids=prompt_ids,
                                neg_prompt_ids=n_prompt_ids, 
                                params=self.params,
                                prng_seed=prng_seed, jit = False,
                                **kwargs
                                ).images

    def process_controlnet_pose(self,
                                video_path,
                                prompt,
                                # neg_prompt,
                                # xT = None,
                                chunk_size=8,
                                motion_field_strength_x: float = 12,
                                motion_field_strength_y: float = 12,
                                t0: int = 44,
                                t1: int = 47,
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

        added_prompt = 'best quality, HD, clay stop-motion, claymation, HQ, masterpiece, art, smooth'
        negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly'
        # negative_prompts=neg_prompt
        video, fps = utils.prepare_video(
            video_path, resolution, self.device, self.dtype, False, output_fps=4)
        control = utils.pre_process_pose(
            video, apply_pose_detect=False)
        f, _, h, w = video.shape

        # self.rng, latents_rng = jax.random.split(self.rng)
        # latents = jax.random.normal(latents_rng, (1, 4, h//8, w//8))
        # latents = latents.repeat(f, 0) #latents.repeat(f, 1, 1, 1)

        result = self.inference(image=control,
                                prompt=prompt + ', ' + added_prompt,
                                # height=h,
                                # width=w,
                                jit=True,
                                negative_prompt=negative_prompts,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                controlnet_conditioning_scale=controlnet_conditioning_scale,
                                # eta=eta,
                                # latents=latents,
                                # output_type='numpy',
                                split_to_chunks=False,
                                chunk_size=chunk_size,
                                motion_field_strength_x=jnp.array(motion_field_strength_x),
                                motion_field_strength_y=jnp.array(motion_field_strength_y),
                                t0=t0,
                                t1=t1,
                                # merging_ratio=merging_ratio,
                                )
        return utils.create_gif(result, fps, path=save_path, watermark=None)


if __name__ == "__main__":
    video_path = "Motion 1"
    model = Model(device='cuda', dtype=jnp.float16)
    result = model.process_controlnet_pose(video_path, "An astronaut dancing in the outer space")