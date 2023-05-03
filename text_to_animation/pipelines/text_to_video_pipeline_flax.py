import warnings
from functools import partial
from typing import Dict, List, Optional, Union
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from flax.jax_utils import unreplicate
from flax import jax_utils
from flax.training.common_utils import shard
from PIL import Image
from transformers import CLIPFeatureExtractor, CLIPTokenizer, FlaxCLIPTextModel
from einops import rearrange, repeat
from diffusers.models import FlaxAutoencoderKL, FlaxControlNetModel, FlaxUNet2DConditionModel
from diffusers.schedulers import (
    FlaxDDIMScheduler,
    FlaxDPMSolverMultistepScheduler,
    FlaxLMSDiscreteScheduler,
    FlaxPNDMScheduler,
)
from diffusers.utils import PIL_INTERPOLATION, logging, replace_example_docstring
from diffusers.pipelines.pipeline_flax_utils import FlaxDiffusionPipeline
from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker_flax import FlaxStableDiffusionSafetyChecker
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
"""
Text2Video-Zero:
 - Inputs: Prompt, Pose Control via mp4/gif, First Frame (?)
 - JAX implementation
 - 3DUnet to replace 2DUnetConditional

"""

def replicate_devices(array):
    return jnp.expand_dims(array, 0).repeat(jax.device_count(), 0)


DEBUG = False # Set to True to use python for loop instead of jax.fori_loop for easier debugging

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import jax
        >>> import numpy as np
        >>> import jax.numpy as jnp
        >>> from flax.jax_utils import replicate
        >>> from flax.training.common_utils import shard
        >>> from diffusers.utils import load_image
        >>> from PIL import Image
        >>> from diffusers import FlaxStableDiffusionControlNetPipeline, FlaxControlNetModel
        >>> def image_grid(imgs, rows, cols):
        ...     w, h = imgs[0].size
        ...     grid = Image.new("RGB", size=(cols * w, rows * h))
        ...     for i, img in enumerate(imgs):
        ...         grid.paste(img, box=(i % cols * w, i // cols * h))
        ...     return grid
        >>> def create_key(seed=0):
        ...     return jax.random.PRNGKey(seed)
        >>> rng = create_key(0)
        >>> # get canny image
        >>> canny_image = load_image(
        ...     "https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/blog_post_cell_10_output_0.jpeg"
        ... )
        >>> prompts = "best quality, extremely detailed"
        >>> negative_prompts = "monochrome, lowres, bad anatomy, worst quality, low quality"
        >>> # load control net and stable diffusion v1-5
        >>> controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
        ...     "lllyasviel/sd-controlnet-canny", from_pt=True, dtype=jnp.float32
        ... )
        >>> pipe, params = FlaxStableDiffusionControlNetPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, revision="flax", dtype=jnp.float32
        ... )
        >>> params["controlnet"] = controlnet_params
        >>> num_samples = jax.device_count()
        >>> rng = jax.random.split(rng, jax.device_count())
        >>> prompt_ids = pipe.prepare_text_inputs([prompts] * num_samples)
        >>> negative_prompt_ids = pipe.prepare_text_inputs([negative_prompts] * num_samples)
        >>> processed_image = pipe.prepare_image_inputs([canny_image] * num_samples)
        >>> p_params = replicate(params)
        >>> prompt_ids = shard(prompt_ids)
        >>> negative_prompt_ids = shard(negative_prompt_ids)
        >>> processed_image = shard(processed_image)
        >>> output = pipe(
        ...     prompt_ids=prompt_ids,
        ...     image=processed_image,
        ...     params=p_params,
        ...     prng_seed=rng,
        ...     num_inference_steps=50,
        ...     neg_prompt_ids=negative_prompt_ids,
        ...     jit=True,
        ... ).images
        >>> output_images = pipe.numpy_to_pil(np.asarray(output.reshape((num_samples,) + output.shape[-3:])))
        >>> output_images = image_grid(output_images, num_samples // 4, 4)
        >>> output_images.save("generated_image.png")
        ```
"""
class FlaxTextToVideoPipeline(FlaxDiffusionPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        unet_vanilla,
        controlnet,
        scheduler: Union[
            FlaxDDIMScheduler, FlaxPNDMScheduler, FlaxLMSDiscreteScheduler, FlaxDPMSolverMultistepScheduler
        ],
        safety_checker: FlaxStableDiffusionSafetyChecker,
        feature_extractor: CLIPFeatureExtractor,
        dtype: jnp.dtype = jnp.float32,
    ):
        super().__init__()
        self.dtype = dtype

        if safety_checker is None:
            logger.warn(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            unet_vanilla=unet_vanilla,
            controlnet=controlnet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def DDPM_forward(self, params, prng, x0, t0, tMax, shape, text_embeddings):
        if x0 is None:
            return jax.random.normal(prng, shape, dtype=text_embeddings.dtype)
        else:
            eps = jax.random.normal(prng, x0.shape, dtype=text_embeddings.dtype)
            alpha_vec = jnp.prod(params["scheduler"].common.alphas[t0:tMax])
            xt = jnp.sqrt(alpha_vec) * x0 + \
                jnp.sqrt(1-alpha_vec) * eps
            return xt
        
    def DDIM_backward(self, params, num_inference_steps, timesteps, skip_t, t0, t1, do_classifier_free_guidance, text_embeddings, latents_local,
                        guidance_scale, controlnet_image=None, controlnet_conditioning_scale=None):
        scheduler_state = self.scheduler.set_timesteps(params["scheduler"], num_inference_steps)
        f = latents_local.shape[2]
        latents_local = rearrange(latents_local, "b c f h w -> (b f) c h w")
        latents = latents_local.copy()
        x_t0_1 = None
        x_t1_1 = None
        max_timestep = len(timesteps)-1
        timesteps = jnp.array(timesteps)
        def while_body(args):
            step, latents, x_t0_1, x_t1_1, scheduler_state = args
            t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
            latent_model_input = jnp.concatenate(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                scheduler_state, latent_model_input, timestep=t
            )
            f = latents.shape[0]
            te = jnp.stack([text_embeddings[0, :, :]]*f + [text_embeddings[-1,:,:]]*f)
            timestep = jnp.broadcast_to(t, latent_model_input.shape[0])
            if controlnet_image is not None:
                down_block_res_samples, mid_block_res_sample = self.controlnet.apply(
                    {"params": params["controlnet"]},
                    jnp.array(latent_model_input),
                    jnp.array(timestep, dtype=jnp.int32),
                    encoder_hidden_states=te,
                    controlnet_cond=controlnet_image,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )
                # predict the noise residual
                noise_pred = self.unet.apply(
                    {"params": params["unet"]},
                    jnp.array(latent_model_input),
                    jnp.array(timestep, dtype=jnp.int32),
                    encoder_hidden_states=te,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            else:
                noise_pred = self.unet.apply(
                    {"params": params["unet"]},
                    jnp.array(latent_model_input),
                    jnp.array(timestep, dtype=jnp.int32),
                    encoder_hidden_states=te,
                    ).sample
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = jnp.split(noise_pred, 2, axis=0)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents, scheduler_state = self.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
            x_t0_1 = jax.lax.select((step < max_timestep-1) & (timesteps[step+1] == t0), latents, x_t0_1)
            x_t1_1 = jax.lax.select((step < max_timestep-1) & (timesteps[step+1] == t1), latents, x_t1_1)
            return (step + 1, latents, x_t0_1, x_t1_1, scheduler_state)
        
        latents_shape = latents.shape
        x_t0_1, x_t1_1 = jnp.zeros(latents_shape), jnp.zeros(latents_shape)

        def cond_fun(arg):
            step, latents, x_t0_1, x_t1_1, scheduler_state = arg
            return (step < skip_t) & (step < num_inference_steps)
        
        if DEBUG:
            step = 0
            while cond_fun((step, latents, x_t0_1, x_t1_1)):
                step, latents, x_t0_1, x_t1_1, scheduler_state = while_body((step, latents, x_t0_1, x_t1_1, scheduler_state))
                step = step + 1
        else:
            _, latents, x_t0_1, x_t1_1, scheduler_state = jax.lax.while_loop(cond_fun, while_body, (0, latents, x_t0_1, x_t1_1, scheduler_state))
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=f)
        res = {"x0": latents.copy()}
        if x_t0_1 is not None:
            x_t0_1 = rearrange(x_t0_1, "(b f) c h w -> b c f  h w", f=f)
            res["x_t0_1"] = x_t0_1.copy()
        if x_t1_1 is not None:
            x_t1_1 = rearrange(x_t1_1, "(b f) c h w -> b c f  h w", f=f)
            res["x_t1_1"] = x_t1_1.copy()
        return res
    
    def warp_latents_independently(self, latents, reference_flow):
        _, _, H, W = reference_flow.shape
        b, _, f, h, w = latents.shape
        assert b == 1
        coords0 = coords_grid(f, H, W)
        coords_t0 = coords0 + reference_flow
        coords_t0 = coords_t0.at[:, 0].set(coords_t0[:, 0] * w / W)
        coords_t0 = coords_t0.at[:, 1].set(coords_t0[:, 1] * h / H)
        f, c, _, _ = coords_t0.shape
        coords_t0 = jax.image.resize(coords_t0, (f, c, h, w), "linear")
        coords_t0 = rearrange(coords_t0, 'f c h w -> f h w c')
        latents_0 = rearrange(latents[0], 'c f h w -> f  c  h w')
        warped = grid_sample(latents_0, coords_t0, "mirror")
        warped = rearrange(warped, '(b f) c h w -> b c f h w', f=f)
        return warped

    def warp_vid_independently(self, vid, reference_flow):
        _, _, H, W = reference_flow.shape
        f, _, h, w = vid.shape
        coords0 = coords_grid(f, H, W)
        coords_t0 = coords0 + reference_flow
        coords_t0 = coords_t0.at[:, 0].set(coords_t0[:, 0] * w / W)
        coords_t0 = coords_t0.at[:, 1].set(coords_t0[:, 1] * h / H)
        f, c, _, _ = coords_t0.shape
        coords_t0 = jax.image.resize(coords_t0, (f, c, h, w), "linear")
        coords_t0 = rearrange(coords_t0, 'f c h w -> f h w c')
        # latents_0 = rearrange(vid, 'c f h w -> f  c  h w')
        warped = grid_sample(vid, coords_t0, "zeropad")
        # warped = rearrange(warped, 'f c h w -> b c f h w', f=f)
        return warped
    
    def create_motion_field(self, motion_field_strength_x, motion_field_strength_y, frame_ids, video_length, latents):
        reference_flow = jnp.zeros(
            (video_length-1, 2, 512, 512), dtype=latents.dtype)    
        for fr_idx, frame_id in enumerate(frame_ids):
            reference_flow = reference_flow.at[fr_idx, 0, :,
                           :].set(motion_field_strength_x*(frame_id))
            reference_flow = reference_flow.at[fr_idx, 1, :,
                           :].set(motion_field_strength_y*(frame_id))
        return reference_flow
    
    def create_motion_field_and_warp_latents(self, motion_field_strength_x, motion_field_strength_y, frame_ids, video_length, latents):
        motion_field = self.create_motion_field(motion_field_strength_x=motion_field_strength_x,
                                                motion_field_strength_y=motion_field_strength_y, latents=latents, video_length=video_length, frame_ids=frame_ids)
        for idx, latent in enumerate(latents):
            latents = latents.at[idx].set(self.warp_latents_independently(
                latent[None], motion_field)[0])
        return motion_field, latents

    def text_to_video_zero(self, params,
                           prng,
                           text_embeddings,
                           video_length: Optional[int],
                           do_classifier_free_guidance = True,
                           height: Optional[int] = None,
                           width: Optional[int] = None,
                           num_inference_steps: int = 50,
                           guidance_scale: float = 7.5,
                           num_videos_per_prompt: Optional[int] = 1,
                           xT = None,
                           smooth_bg_strength: float=0.,
                           motion_field_strength_x: float = 12,
                           motion_field_strength_y: float = 12,
                           t0: int = 44,
                           t1: int = 47,
                           controlnet_image=None,
                           controlnet_conditioning_scale=0,
                           ):
        frame_ids = list(range(video_length))
        # Prepare timesteps
        params["scheduler"] = self.scheduler.set_timesteps(params["scheduler"], num_inference_steps)
        timesteps = params["scheduler"].timesteps
        # Prepare latent variables
        num_channels_latents = self.unet.in_channels
        batch_size = 1
        xT = prepare_latents(params, prng, batch_size * num_videos_per_prompt, num_channels_latents, height, width, self.vae_scale_factor, xT)

        timesteps_ddpm = [981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741, 721,
                            701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461, 441,
                            421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181, 161,
                            141, 121, 101,  81,  61,  41,  21,   1]
        timesteps_ddpm.reverse()
        t0 = timesteps_ddpm[t0]
        t1 = timesteps_ddpm[t1]
        x_t1_1 = None

        # Denoising loop
        shape = (batch_size, num_channels_latents, 1, height //
                self.vae.scaling_factor, width // self.vae.scaling_factor)

        #  perform ∆t backward steps by stable diffusion
        ddim_res = self.DDIM_backward(params, num_inference_steps=num_inference_steps, timesteps=timesteps, skip_t=1000, t0=t0, t1=t1, do_classifier_free_guidance=do_classifier_free_guidance,
                                text_embeddings=text_embeddings, latents_local=xT, guidance_scale=guidance_scale,
                                controlnet_image=jnp.stack([controlnet_image[0]] * 2), controlnet_conditioning_scale=controlnet_conditioning_scale)
        x0 = ddim_res["x0"]

        # apply warping functions
        if "x_t0_1" in ddim_res:
            x_t0_1 = ddim_res["x_t0_1"]
        if "x_t1_1" in ddim_res:
            x_t1_1 = ddim_res["x_t1_1"]
        x_t0_k = x_t0_1[:, :, :1, :, :].repeat(video_length-1, 2)
        reference_flow, x_t0_k = self.create_motion_field_and_warp_latents(
            motion_field_strength_x=motion_field_strength_x, motion_field_strength_y=motion_field_strength_y, latents=x_t0_k, video_length=video_length, frame_ids=frame_ids[1:])
        # assuming t0=t1=1000, if t0 = 1000

        # DDPM forward for more motion freedom
        ddpm_fwd = partial(self.DDPM_forward, params=params, prng=prng, x0=x_t0_k, t0=t0,
                           tMax=t1, shape=shape, text_embeddings=text_embeddings)
        x_t1_k = jax.lax.cond(t1 > t0,
                              ddpm_fwd,
                              lambda:x_t0_k
        )
        x_t1 = jnp.concatenate([x_t1_1, x_t1_k], axis=2)

        # backward stepts by stable diffusion

        #warp the controlnet image following the same flow defined for latent
        controlnet_video = controlnet_image[:video_length]
        controlnet_video = controlnet_video.at[1:].set(self.warp_vid_independently(controlnet_video[1:], reference_flow))
        controlnet_image = jnp.concatenate([controlnet_video]*2)
        smooth_bg = True

        if smooth_bg:
            #latent shape: "b c f h w"
            M_FG = repeat(get_mask_pose(controlnet_video), "f h w -> b c f h w", c=x_t1.shape[1], b=batch_size)
            initial_bg = repeat(x_t1[:,:,0] * (1 - M_FG[:,:,0]), "b c h w -> b c f h w", f=video_length-1)
            #warp the controlnet image following the same flow defined for latent #f c h w
            initial_bg_warped = self.warp_latents_independently(initial_bg, reference_flow)
            bgs = x_t1[:,:,1:] * (1 - M_FG[:,:,1:]) #initial background 
            initial_mask_warped = 1 - self.warp_latents_independently(repeat(M_FG[:,:,0], "b c h w -> b c f h w", f = video_length-1), reference_flow)
            # initial_mask_warped = 1 - warp_vid_independently(repeat(M_FG[:,:,0], "b c h w -> (b f) c h w", f = video_length-1), reference_flow)
            # initial_mask_warped = rearrange(initial_mask_warped, "(b f) c h w -> b c f h w", b=batch_size)
            mask = (1 - M_FG[:,:,1:]) * initial_mask_warped
            x_t1 = x_t1.at[:,:,1:].set( (1 - mask) * x_t1[:,:,1:] + mask * (initial_bg_warped * smooth_bg_strength + (1 - smooth_bg_strength) * bgs))
            
        ddim_res = self.DDIM_backward(params, num_inference_steps=num_inference_steps, timesteps=timesteps, skip_t=t1, t0=-1, t1=-1, do_classifier_free_guidance=do_classifier_free_guidance,
                                            text_embeddings=text_embeddings, latents_local=x_t1, guidance_scale=guidance_scale,
                                            controlnet_image=controlnet_image, controlnet_conditioning_scale=controlnet_conditioning_scale,
                                     )
        
        x0 = ddim_res["x0"]
        del ddim_res
        del x_t1
        del x_t1_1
        del x_t1_k
        return x0

    def denoise_latent(self, params, num_inference_steps, timesteps, do_classifier_free_guidance, text_embeddings, latents,
                        guidance_scale, controlnet_image=None, controlnet_conditioning_scale=None):
        
        scheduler_state = self.scheduler.set_timesteps(params["scheduler"], num_inference_steps)
        # f = latents_local.shape[2]
        # latents_local = rearrange(latents_local, "b c f h w -> (b f) c h w")

        max_timestep = len(timesteps)-1
        timesteps = jnp.array(timesteps)
        def while_body(args):
            step, latents, scheduler_state = args
            t = jnp.array(scheduler_state.timesteps, dtype=jnp.int32)[step]
            latent_model_input = jnp.concatenate(
                [latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(
                scheduler_state, latent_model_input, timestep=t
            )
            f = latents.shape[0]
            te = jnp.stack([text_embeddings[0, :, :]]*f + [text_embeddings[-1,:,:]]*f)
            timestep = jnp.broadcast_to(t, latent_model_input.shape[0])
            if controlnet_image is not None:
                down_block_res_samples, mid_block_res_sample = self.controlnet.apply(
                    {"params": params["controlnet"]},
                    jnp.array(latent_model_input),
                    jnp.array(timestep, dtype=jnp.int32),
                    encoder_hidden_states=te,
                    controlnet_cond=controlnet_image,
                    conditioning_scale=controlnet_conditioning_scale,
                    return_dict=False,
                )
                # predict the noise residual
                noise_pred = self.unet_vanilla.apply(
                    {"params": params["unet"]},
                    jnp.array(latent_model_input),
                    jnp.array(timestep, dtype=jnp.int32),
                    encoder_hidden_states=te,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
            else:
                noise_pred = self.unet_vanilla.apply(
                    {"params": params["unet"]},
                    jnp.array(latent_model_input),
                    jnp.array(timestep, dtype=jnp.int32),
                    encoder_hidden_states=te,
                    ).sample
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = jnp.split(noise_pred, 2, axis=0)
                noise_pred = noise_pred_uncond + guidance_scale * \
                    (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents, scheduler_state = self.scheduler.step(scheduler_state, noise_pred, t, latents).to_tuple()
            return (step + 1, latents, scheduler_state)
        
        def cond_fun(arg):
            step, latents, scheduler_state = arg
            return (step < num_inference_steps)
        
        if DEBUG:
            step = 0
            while cond_fun((step, latents, scheduler_state)):
                step, latents, scheduler_state = while_body((step, latents, scheduler_state))
                step = step + 1
        else:
            _, latents, scheduler_state = jax.lax.while_loop(cond_fun, while_body, (0, latents, scheduler_state))
        # latents = rearrange(latents, "(b f) c h w -> b c f h w", f=f)
        return latents

    def generate_starting_frames(self,
                                params,
                                prngs: list, #list of prngs for each img
                                prompt,
                                neg_prompt,
                                controlnet_image,
                                do_classifier_free_guidance = True,
                                num_inference_steps: int = 50,
                                guidance_scale: float = 7.5,
                                t0: int = 44,
                                t1: int = 47,
                                controlnet_conditioning_scale=1.,
                                ):

        height, width = controlnet_image.shape[-2:]
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 64 but are {height} and {width}.")

        shape = (self.unet.in_channels, height //
        self.vae_scale_factor, width // self.vae_scale_factor) #b c h w
        # scale the initial noise by the standard deviation required by the scheduler

        print(f"Generating {len(prngs)} first frames with prompt {prompt}, for {num_inference_steps} steps. PRNG seeds are: {prngs}")

        latents = jnp.stack([jax.random.normal(prng, shape) for prng in prngs]) # b c h w
        latents = latents * params["scheduler"].init_noise_sigma

        timesteps = params["scheduler"].timesteps
        timesteps_ddpm = [981, 961, 941, 921, 901, 881, 861, 841, 821, 801, 781, 761, 741, 721,
                            701, 681, 661, 641, 621, 601, 581, 561, 541, 521, 501, 481, 461, 441,
                            421, 401, 381, 361, 341, 321, 301, 281, 261, 241, 221, 201, 181, 161,
                            141, 121, 101,  81,  61,  41,  21,   1]
        timesteps_ddpm.reverse()
        t0 = timesteps_ddpm[t0]
        t1 = timesteps_ddpm[t1]

        # get prompt text embeddings
        prompt_ids = self.prepare_text_inputs(prompt)
        prompt_embeds = self.text_encoder(prompt_ids, params=params["text_encoder"])[0]

        # TODO: currently it is assumed `do_classifier_free_guidance = guidance_scale > 1.0`
        # implement this conditional `do_classifier_free_guidance = guidance_scale > 1.0`
        batch_size = 1
        max_length = prompt_ids.shape[-1]
        if neg_prompt is None:
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="np"
            ).input_ids
        else:
            neg_prompt_ids = self.prepare_text_inputs(neg_prompt)
            uncond_input = neg_prompt_ids

        negative_prompt_embeds = self.text_encoder(uncond_input, params=params["text_encoder"])[0]
        text_embeddings = jnp.concatenate([negative_prompt_embeds, prompt_embeds])
        controlnet_image = jnp.stack([controlnet_image[0]] * 2 * len(prngs))

        @jax.jit
        def _generate_starting_frames(params, text_embeddings, latents, controlnet_image, controlnet_conditioning_scale):
            #  perform ∆t backward steps by stable diffusion

            # delta_t_diffusion = jax.vmap(lambda latent : self.DDIM_backward(params, num_inference_steps=num_inference_steps, timesteps=timesteps, skip_t=1000, t0=t0, t1=t1, do_classifier_free_guidance=do_classifier_free_guidance,
            #                                     text_embeddings=text_embeddings, latents_local=latent, guidance_scale=guidance_scale,
            #                                     controlnet_image=controlnet_image, controlnet_conditioning_scale=controlnet_conditioning_scale))


            # ddim_res = delta_t_diffusion(latents)
            # latents = ddim_res["x0"] #output is  i b c f h w

            # DDPM forward for more motion freedom
            # ddpm_fwd = jax.vmap(lambda prng, latent: self.DDPM_forward(params=params, prng=prng, x0=latent, t0=t0,
            #                 tMax=t1, shape=shape, text_embeddings=text_embeddings))

            # latents = ddpm_fwd(stacked_prngs, latents)

            # main backward diffusion
            # denoise_first_frame = lambda latent : self.DDIM_backward(params, num_inference_steps=num_inference_steps, timesteps=timesteps, skip_t=100000, t0=-1, t1=-1, do_classifier_free_guidance=do_classifier_free_guidance,
            #                                     text_embeddings=text_embeddings, latents_local=latent, guidance_scale=guidance_scale,
            #                                     controlnet_image=controlnet_image, controlnet_conditioning_scale=controlnet_conditioning_scale, use_vanilla=True)
            # latents = rearrange(latents, 'i b c f h w -> (i b) c f h w')
            # ddim_res = denoise_first_frame(latents)
            latents = self.denoise_latent(params, num_inference_steps=num_inference_steps, timesteps=timesteps, do_classifier_free_guidance=do_classifier_free_guidance,
                                                text_embeddings=text_embeddings, latents=latents, guidance_scale=guidance_scale,
                                                controlnet_image=controlnet_image, controlnet_conditioning_scale=controlnet_conditioning_scale)
            # latents = rearrange(ddim_res["x0"], 'i b c f h w -> (i b) c f h w') #output is  i b c f h w

            # scale and decode the image latents with vae
            latents = 1 / self.vae.config.scaling_factor * latents
            # latents = rearrange(latents, "b c h w -> (b f) c h w")
            imgs = self.vae.apply({"params": params["vae"]}, latents, method=self.vae.decode).sample
            imgs = (imgs / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
            return imgs
        
        return _generate_starting_frames(params, text_embeddings, latents, controlnet_image, controlnet_conditioning_scale)

    def generate_video(
        self,
        prompt: str,
        image: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.random.KeyArray,
        num_inference_steps: int = 50,
        guidance_scale: Union[float, jnp.array] = 7.5,
        latents: jnp.array = None,
        neg_prompt: str = "",
        controlnet_conditioning_scale: Union[float, jnp.array] = 1.0,
        return_dict: bool = True,
        jit: bool = False,
        xT = None,
        smooth_bg_strength: float=0.,
        motion_field_strength_x: float = 3,
        motion_field_strength_y: float = 4,
        t0: int = 44,
        t1: int = 47,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt_ids (`jnp.array`):
                The prompt or prompts to guide the image generation.
            image (`jnp.array`):
                Array representing the ControlNet input condition. ControlNet use this input condition to generate
                guidance to Unet.
            params (`Dict` or `FrozenDict`): Dictionary containing the model parameters/weights
            prng_seed (`jax.random.KeyArray` or `jax.Array`): Array containing random number generator key
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            latents (`jnp.array`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            controlnet_conditioning_scale (`float` or `jnp.array`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] instead of
                a plain tuple.
            jit (`bool`, defaults to `False`):
                Whether to run `pmap` versions of the generation and safety scoring functions. NOTE: This argument
                exists because `__call__` is not yet end-to-end pmap-able. It will be removed in a future release.
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple. When returning a tuple, the first element is a list with the generated images, and the second
            element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        height, width = image.shape[-2:]
        vid_length = image.shape[0]
        # get prompt text embeddings
        prompt_ids = self.prepare_text_inputs([prompt] * vid_length)
        neg_prompt_ids = self.prepare_text_inputs([neg_prompt] * vid_length)

        # TODO: currently it is assumed `do_classifier_free_guidance = guidance_scale > 1.0`
        # implement this conditional `do_classifier_free_guidance = guidance_scale > 1.0`
        batch_size = 1

        if isinstance(guidance_scale, float):
            # Convert to a tensor so each device gets a copy. Follow the prompt_ids for
            # shape information, as they may be sharded (when `jit` is `True`), or not.
            guidance_scale = jnp.array([guidance_scale] * prompt_ids.shape[0])
            if len(prompt_ids.shape) > 2:
                # Assume sharded
                guidance_scale = guidance_scale[:, None]
        if isinstance(controlnet_conditioning_scale, float):
            # Convert to a tensor so each device gets a copy. Follow the prompt_ids for
            # shape information, as they may be sharded (when `jit` is `True`), or not.
            controlnet_conditioning_scale = jnp.array([controlnet_conditioning_scale] * prompt_ids.shape[0])
            if len(prompt_ids.shape) > 2:
                # Assume sharded
                controlnet_conditioning_scale = controlnet_conditioning_scale[:, None]
        if jit:
            images = _p_generate(
                self,
                replicate_devices(prompt_ids),
                replicate_devices(image),
                jax_utils.replicate(params),
                replicate_devices(prng_seed),
                num_inference_steps,
                replicate_devices(guidance_scale),
                replicate_devices(latents) if latents is not None else None,
                replicate_devices(neg_prompt_ids) if neg_prompt_ids is not None else None,
                replicate_devices(controlnet_conditioning_scale),
                replicate_devices(xT) if xT is not None else None,
                replicate_devices(smooth_bg_strength),
                replicate_devices(motion_field_strength_x),
                replicate_devices(motion_field_strength_y),
                t0,
                t1,
            )
        else:
            images = self._generate(
                prompt_ids,
                image,
                params,
                prng_seed,
                num_inference_steps,
                guidance_scale,
                latents,
                neg_prompt_ids,
                controlnet_conditioning_scale,
                xT,
                smooth_bg_strength,
                motion_field_strength_x,
                motion_field_strength_y,
                t0,
                t1,
            )
        if self.safety_checker is not None:
            safety_params = params["safety_checker"]
            images_uint8_casted = (images * 255).round().astype("uint8")
            num_devices, batch_size = images.shape[:2]
            images_uint8_casted = np.asarray(images_uint8_casted).reshape(num_devices * batch_size, height, width, 3)
            images_uint8_casted, has_nsfw_concept = self._run_safety_checker(images_uint8_casted, safety_params, jit)
            images = np.asarray(images)
            # block images
            if any(has_nsfw_concept):
                for i, is_nsfw in enumerate(has_nsfw_concept):
                    if is_nsfw:
                        images[i] = np.asarray(images_uint8_casted[i])
            images = images.reshape(num_devices, batch_size, height, width, 3)
        else:
            images = np.asarray(images)
            has_nsfw_concept = False
        if not return_dict:
            return (images, has_nsfw_concept)
        return FlaxStableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw_concept)

    def prepare_text_inputs(self, prompt: Union[str, List[str]]):
        if not isinstance(prompt, (str, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        return text_input.input_ids
    def prepare_image_inputs(self, image: Union[Image.Image, List[Image.Image]]):
        if not isinstance(image, (Image.Image, list)):
            raise ValueError(f"image has to be of type `PIL.Image.Image` or list but is {type(image)}")
        if isinstance(image, Image.Image):
            image = [image]
        processed_images = jnp.concatenate([preprocess(img, jnp.float32) for img in image])
        return processed_images
    def _get_has_nsfw_concepts(self, features, params):
        has_nsfw_concepts = self.safety_checker(features, params)
        return has_nsfw_concepts
    def _run_safety_checker(self, images, safety_model_params, jit=False):
        # safety_model_params should already be replicated when jit is True
        pil_images = [Image.fromarray(image) for image in images]
        features = self.feature_extractor(pil_images, return_tensors="np").pixel_values
        if jit:
            features = shard(features)
            has_nsfw_concepts = _p_get_has_nsfw_concepts(self, features, safety_model_params)
            has_nsfw_concepts = unshard(has_nsfw_concepts)
            safety_model_params = unreplicate(safety_model_params)
        else:
            has_nsfw_concepts = self._get_has_nsfw_concepts(features, safety_model_params)
        images_was_copied = False
        for idx, has_nsfw_concept in enumerate(has_nsfw_concepts):
            if has_nsfw_concept:
                if not images_was_copied:
                    images_was_copied = True
                    images = images.copy()
                images[idx] = np.zeros(images[idx].shape, dtype=np.uint8)  # black image
            if any(has_nsfw_concepts):
                warnings.warn(
                    "Potential NSFW content was detected in one or more images. A black image will be returned"
                    " instead. Try again with a different prompt and/or seed."
                )
        return images, has_nsfw_concepts
    def _generate(
        self,
        prompt_ids: jnp.array,
        image: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.random.KeyArray,
        num_inference_steps: int,
        guidance_scale: float,
        latents: Optional[jnp.array] = None,
        neg_prompt_ids: Optional[jnp.array] = None,
        controlnet_conditioning_scale: float = 1.0,
        xT = None,
        smooth_bg_strength: float = 0.,
        motion_field_strength_x: float = 12,
        motion_field_strength_y: float = 12,
        t0: int = 44,
        t1: int = 47,
    ):
        height, width = image.shape[-2:]
        video_length = image.shape[0]
        if height % 64 != 0 or width % 64 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 64 but are {height} and {width}.")
        # get prompt text embeddings
        prompt_embeds = self.text_encoder(prompt_ids, params=params["text_encoder"])[0]
        # TODO: currently it is assumed `do_classifier_free_guidance = guidance_scale > 1.0`
        # implement this conditional `do_classifier_free_guidance = guidance_scale > 1.0`
        batch_size = prompt_ids.shape[0]
        max_length = prompt_ids.shape[-1]
        if neg_prompt_ids is None:
            uncond_input = self.tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="np"
            ).input_ids
        else:
            uncond_input = neg_prompt_ids
        negative_prompt_embeds = self.text_encoder(uncond_input, params=params["text_encoder"])[0]
        context = jnp.concatenate([negative_prompt_embeds, prompt_embeds])
        image = jnp.concatenate([image] * 2)
        seed_t2vz, prng_seed = jax.random.split(prng_seed)
        #get the latent following text to video zero
        latents = self.text_to_video_zero(params, seed_t2vz, text_embeddings=context, video_length=video_length,
                                          height=height, width = width, num_inference_steps=num_inference_steps,
                                          guidance_scale=guidance_scale, controlnet_image=image,
                                          xT=xT, smooth_bg_strength=smooth_bg_strength, t0=t0, t1=t1,
                                          motion_field_strength_x=motion_field_strength_x,
                                          motion_field_strength_y=motion_field_strength_y,
                                          controlnet_conditioning_scale=controlnet_conditioning_scale
                                          )
        # scale and decode the image latents with vae
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = self.vae.apply({"params": params["vae"]}, latents, method=self.vae.decode).sample
        video = (video / 2 + 0.5).clip(0, 1).transpose(0, 2, 3, 1)
        return video
    
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt_ids: jnp.array,
        image: jnp.array,
        params: Union[Dict, FrozenDict],
        prng_seed: jax.random.KeyArray,
        num_inference_steps: int = 50,
        guidance_scale: Union[float, jnp.array] = 7.5,
        latents: jnp.array = None,
        neg_prompt_ids: jnp.array = None,
        controlnet_conditioning_scale: Union[float, jnp.array] = 1.0,
        return_dict: bool = True,
        jit: bool = False,
        xT = None,
        smooth_bg_strength: float = 0.,
        motion_field_strength_x: float = 3,
        motion_field_strength_y: float = 4,
        t0: int = 44,
        t1: int = 47,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt_ids (`jnp.array`):
                The prompt or prompts to guide the image generation.
            image (`jnp.array`):
                Array representing the ControlNet input condition. ControlNet use this input condition to generate
                guidance to Unet.
            params (`Dict` or `FrozenDict`): Dictionary containing the model parameters/weights
            prng_seed (`jax.random.KeyArray` or `jax.Array`): Array containing random number generator key
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            latents (`jnp.array`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            controlnet_conditioning_scale (`float` or `jnp.array`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] instead of
                a plain tuple.
            jit (`bool`, defaults to `False`):
                Whether to run `pmap` versions of the generation and safety scoring functions. NOTE: This argument
                exists because `__call__` is not yet end-to-end pmap-able. It will be removed in a future release.
        Examples:
        Returns:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple. When returning a tuple, the first element is a list with the generated images, and the second
            element is a list of `bool`s denoting whether the corresponding generated image likely represents
            "not-safe-for-work" (nsfw) content, according to the `safety_checker`.
        """
        height, width = image.shape[-2:]
        if isinstance(guidance_scale, float):
            # Convert to a tensor so each device gets a copy. Follow the prompt_ids for
            # shape information, as they may be sharded (when `jit` is `True`), or not.
            guidance_scale = jnp.array([guidance_scale] * prompt_ids.shape[0])
            if len(prompt_ids.shape) > 2:
                # Assume sharded
                guidance_scale = guidance_scale[:, None]
        if isinstance(controlnet_conditioning_scale, float):
            # Convert to a tensor so each device gets a copy. Follow the prompt_ids for
            # shape information, as they may be sharded (when `jit` is `True`), or not.
            controlnet_conditioning_scale = jnp.array([controlnet_conditioning_scale] * prompt_ids.shape[0])
            if len(prompt_ids.shape) > 2:
                # Assume sharded
                controlnet_conditioning_scale = controlnet_conditioning_scale[:, None]
        if jit:
            images = _p_generate(
                self,
                prompt_ids,
                image,
                params,
                prng_seed,
                num_inference_steps,
                guidance_scale,
                latents,
                neg_prompt_ids,
                controlnet_conditioning_scale,
                xT,
                smooth_bg_strength,
                motion_field_strength_x,
                motion_field_strength_y,
                t0,
                t1,
            )
        else:
            images = self._generate(
                prompt_ids,
                image,
                params,
                prng_seed,
                num_inference_steps,
                guidance_scale,
                latents,
                neg_prompt_ids,
                controlnet_conditioning_scale,
                xT,
                smooth_bg_strength,
                motion_field_strength_x,
                motion_field_strength_y,
                t0,
                t1,
            )
        if self.safety_checker is not None:
            safety_params = params["safety_checker"]
            images_uint8_casted = (images * 255).round().astype("uint8")
            num_devices, batch_size = images.shape[:2]
            images_uint8_casted = np.asarray(images_uint8_casted).reshape(num_devices * batch_size, height, width, 3)
            images_uint8_casted, has_nsfw_concept = self._run_safety_checker(images_uint8_casted, safety_params, jit)
            images = np.asarray(images)
            # block images
            if any(has_nsfw_concept):
                for i, is_nsfw in enumerate(has_nsfw_concept):
                    if is_nsfw:
                        images[i] = np.asarray(images_uint8_casted[i])
            images = images.reshape(num_devices, batch_size, height, width, 3)
        else:
            images = np.asarray(images)
            has_nsfw_concept = False
        if not return_dict:
            return (images, has_nsfw_concept)
        return FlaxStableDiffusionPipelineOutput(images=images, nsfw_content_detected=has_nsfw_concept)


# Static argnums are pipe, num_inference_steps. A change would trigger recompilation.
# Non-static args are (sharded) input tensors mapped over their first dimension (hence, `0`).
@partial(
    jax.pmap,
    in_axes=(None, 0, 0, 0, 0, None, 0, 0, 0, 0, 0, 0, 0, 0, None, None),
    static_broadcasted_argnums=(0, 5, 14, 15)
)
def _p_generate(
    pipe,
    prompt_ids, 
    image,
    params,
    prng_seed,
    num_inference_steps,
    guidance_scale,
    latents,
    neg_prompt_ids,
    controlnet_conditioning_scale,
    xT,
    smooth_bg_strength,
    motion_field_strength_x,
    motion_field_strength_y,
    t0,
    t1,
):
    return pipe._generate(
        prompt_ids,
        image,
        params,
        prng_seed,
        num_inference_steps,
        guidance_scale,
        latents,
        neg_prompt_ids,
        controlnet_conditioning_scale,
        xT,
        smooth_bg_strength,
        motion_field_strength_x,
        motion_field_strength_y,
        t0,
        t1,
    )
@partial(jax.pmap, static_broadcasted_argnums=(0,))
def _p_get_has_nsfw_concepts(pipe, features, params):
    return pipe._get_has_nsfw_concepts(features, params)
def unshard(x: jnp.ndarray):
    # einops.rearrange(x, 'd b ... -> (d b) ...')
    num_devices, batch_size = x.shape[:2]
    rest = x.shape[2:]
    return x.reshape(num_devices * batch_size, *rest)
def preprocess(image, dtype):
    image = image.convert("RGB")
    w, h = image.size
    w, h = (x - x % 64 for x in (w, h))  # resize to integer multiple of 64
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = jnp.array(image).astype(dtype) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return image

def prepare_latents(params, prng, batch_size, num_channels_latents, height, width, vae_scale_factor, latents=None):
    shape = (batch_size, num_channels_latents, 1, height //
            vae_scale_factor, width // vae_scale_factor) #b c f h w
    # scale the initial noise by the standard deviation required by the scheduler
    if latents is None:
        latents = jax.random.normal(prng, shape)
    latents = latents * params["scheduler"].init_noise_sigma
    return latents

def coords_grid(batch, ht, wd):
    coords = jnp.meshgrid(jnp.arange(ht), jnp.arange(wd), indexing="ij")
    coords = jnp.stack(coords[::-1], axis=0)
    return coords[None].repeat(batch, 0)

def adapt_pos_mirror(x, y, W, H):
  #adapt the position, with mirror padding
  x_w_mirror = ((x + W - 1) % (2*(W - 1))) - W + 1
  x_adapted = jnp.where(x_w_mirror > 0, x_w_mirror, - (x_w_mirror))
  y_w_mirror = ((y + H - 1) % (2*(H - 1))) - H + 1
  y_adapted = jnp.where(y_w_mirror > 0, y_w_mirror, - (y_w_mirror))
  return y_adapted, x_adapted

def safe_get_zeropad(img, x,y,W,H):
  return jnp.where((x < W) & (x > 0) & (y < H) & (y > 0), img[y,x], 0.)

def safe_get_mirror(img, x,y,W,H):
  return img[adapt_pos_mirror(x,y,W,H)]

@partial(jax.vmap, in_axes=(0, 0, None))
@partial(jax.vmap, in_axes=(0, None, None))
@partial(jax.vmap, in_axes=(None,0, None))
@partial(jax.vmap, in_axes=(None, 0, None))
def grid_sample(latents, grid, method):
    # this is an alternative to torch.functional.nn.grid_sample in jax
    # this implementation is following the algorithm described @ https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    # but with coordinates scaled to the size of the image
    if method == "mirror":
      return safe_get_mirror(latents, jnp.array(grid[0], dtype=jnp.int16), jnp.array(grid[1], dtype=jnp.int16), latents.shape[0], latents.shape[1])
    else: #default is zero padding
      return safe_get_zeropad(latents, jnp.array(grid[0], dtype=jnp.int16), jnp.array(grid[1], dtype=jnp.int16), latents.shape[0], latents.shape[1])

def bandw_vid(vid, threshold):
  vid = jnp.max(vid, axis=1)
  return jnp.where(vid > threshold, 1, 0)

def mean_blur(vid, k):
  window = jnp.ones((vid.shape[0], k, k))/ (k*k)
  convolve=jax.vmap(lambda img, kernel:jax.scipy.signal.convolve(img, kernel, mode='same'))
  smooth_vid = convolve(vid, window)
  return smooth_vid

def get_mask_pose(vid):
  vid = bandw_vid(vid, 0.4)
  l, h, w = vid.shape
  vid = jax.image.resize(vid, (l, h//8, w//8), "nearest")
  vid=bandw_vid(mean_blur(vid, 7)[:,None], threshold=0.01)
  return vid/(jnp.max(vid) + 1e-4)
  #return jax.image.resize(vid/(jnp.max(vid) + 1e-4), (l, h, w), "nearest")