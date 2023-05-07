import gradio as gr
from text_to_animation.model import ControlAnimationModel
import os
from utils.hf_utils import get_model_list
from utils import gradio_utils, utils

from transformers import CLIPTokenizer, CLIPFeatureExtractor, FlaxCLIPTextModel
from diffusers import (
    FlaxDDIMScheduler,
    FlaxAutoencoderKL,
    FlaxStableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
)
from text_to_animation.models.unet_2d_condition_flax import FlaxUNet2DConditionModel
from text_to_animation.models.controlnet_flax import FlaxControlNetModel

import jax
import jax.numpy as jnp
from text_to_animation.model import ControlAnimationModel

model = ControlAnimationModel(dtype=jnp.float16)

huggingspace_name = os.environ.get("SPACE_AUTHOR_NAME")
on_huggingspace = huggingspace_name if huggingspace_name is not None else False

examples = [
    ["an astronaut waving the arm on the moon"],
    ["a sloth surfing on a wakeboard"],
    ["an astronaut walking on a street"],
    ["a cute cat walking on grass"],
    ["a horse is galloping on a street"],
    ["an astronaut is skiing down the hill"],
    ["a gorilla walking alone down the street"],
    ["a gorilla dancing on times square"],
    ["A panda dancing dancing like crazy on Times Square"],
]


gen_images = []  # str path of generated images
gen_seeds = [] # list of seeds used to generate the images
initial_frame = None


#init model

model_id="runwayml/stable-diffusion-v1-5"
controlnet_id = "fusing/stable-diffusion-v1-5-controlnet-openpose"
controlnet, controlnet_params = FlaxControlNetModel.from_pretrained(
    controlnet_id,
    # revision=args.controlnet_revision,
    from_pt=True,
    dtype=jnp.float16,
)
tokenizer = CLIPTokenizer.from_pretrained(
model_id, subfolder="tokenizer", revision="fp16"
)
scheduler, scheduler_state = FlaxDDIMScheduler.from_pretrained(
model_id, subfolder ="scheduler", revision="fp16"
)
model.set_model(
                model_id=model_id,
                tokenizer=tokenizer,
                controlnet=controlnet,
                controlnet_params=controlnet_params,
                scheduler=scheduler,
                scheduler_state=scheduler_state)

def generate_initial_frames(prompt, num_imgs=4, video_path="Motion 1", resolution=512):
    video_path = gradio_utils.motion_to_video_path(
        video_path) if 'Motion' in video_path else video_path

    #added_prompt = 'best quality, HD, clay stop-motion, claymation, HQ, masterpiece, art, smooth'
    #added_prompt = 'high quality, anatomically correct, clay stop-motion, aardman, claymation, smooth'
    negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly'
    video, fps = utils.prepare_video(
        video_path, resolution, None, model.dtype, False, output_fps=4)
    control = utils.pre_process_pose(video, apply_pose_detect=False)
    f, _, h, w = video.shape

    images, seeds = model.generate_starting_frames(
                            control,
                            prompt,
                            negative_prompts,
                            num_imgs=num_imgs,
                            )
    global gen_images
    global gen_seeds
    gen_images = images
    gen_seeds = seeds
    return images

def generate_video_frames(prompt, prng, video_path="Motion 1", lora_scale=0. , resolution=512):

    video_path = gradio_utils.motion_to_video_path(
        video_path) if 'Motion' in video_path else video_path

    #added_prompt = 'best quality, HD, clay stop-motion, claymation, HQ, masterpiece, art, smooth'
    #added_prompt = 'high quality, anatomically correct, clay stop-motion, aardman, claymation, smooth'
    negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer difits, cropped, worst quality, low quality, deformed body, bloated, ugly'
    video, fps = utils.prepare_video(
        video_path, resolution, None, model.dtype, False, output_fps=4)
    control = utils.pre_process_pose(video, apply_pose_detect=False)
    f, _, h, w = video.shape

    images = model.generate_video_from_frame(control, prompt, prng, lora_scale)
    return images



def select_initial_frame(evt: gr.SelectData):
    global initial_frame

    if evt.index < len(gen_images):
        initial_frame = gen_images[evt.index]
        
        print(initial_frame)
        return gr.update(value=gen_seeds[evt.index])

def pose_gallery_callback(evt: gr.SelectData):
    return f"Motion {evt.index+1}"

def on_video_path_update(evt: gr.EventData):
    return f'Selection: **{evt._data}**'

def create_demo():
    # global animation_model

    with gr.Blocks() as demo:
        with gr.Column(visible=True) as frame_selection_col:
            with gr.Row():
                with gr.Column():
                    frames_prompt = gr.Textbox(
                        placeholder="Prompt", show_label=False, lines=4
                    )
                    frames_n_prompt = gr.Textbox(
                        placeholder="Negative Prompt (optional)",
                        show_label=False,
                        lines=2,
                    )

                with gr.Column():
                    # model_link = gr.Textbox(
                    #     label="Model Link",
                    #     placeholder="dreamlike-art/dreamlike-photoreal-2.0",
                    #     info="Give the hugging face model name or URL link to safetensor.",
                    # )
                    # is_safetensor = gr.Checkbox(label="Safetensors")
                    gen_frames_button = gr.Button(
                        value="Generate Initial Frames", variant="primary"
                    )

            with gr.Row():
                with gr.Column(scale=2):
                    # width = gr.Slider(32, 2048, value=512, label="Width")
                    # height = gr.Slider(32, 2048, value=512, label="Height")
                    # cfg_scale = gr.Slider(1, 20, value=7.0, step=0.1, label="CFG scale")
                    gallery_pose_sequence = gr.Gallery(label="Pose Sequence", value=[('__assets__/dance1.gif', "Motion 1"), ('__assets__/dance2.gif', "Motion 2"), (
                    '__assets__/dance3.gif', "Motion 3"), ('__assets__/dance4.gif', "Motion 4"), ('__assets__/dance5.gif', "Motion 5")]).style(grid=[2], height="auto")
                    input_video_path = gr.Textbox(
                        label="Pose Sequence", visible=False, value="Motion 1")
                    gr.Markdown("## Selection")
                    pose_sequence_selector = gr.Markdown(
                        'Pose Sequence: **Motion 1**')
                    lora_scale = gr.Slide(0, 1, value=0, step=0.1, label="Scale of lora model")
                    num_imgs = gr.Slider(1, 8, value=4, step=1, label="Number of images to generate")
                    seed = gr.Slider(
                        label="Seed",
                        info="-1 for random seed on each run. Otherwise, the seed will be fixed.",
                        minimum=-1,
                        maximum=65536,
                        value=0,
                        step=1,
                    )
                    gallery_pose_sequence.select(pose_gallery_callback, None, input_video_path)
                    input_video_path.change(on_video_path_update,
                                None, pose_sequence_selector)

                with gr.Column(scale=3):
                    initial_frames = gr.Gallery(
                        label="Initial Frames", show_label=False
                    ).style(
                        columns=[2], rows=[2], object_fit="scale-down", height="auto"
                    )
                    initial_frames.select(select_initial_frame, None, [seed])
                    select_frame_button = gr.Button(
                        value="Select Initial Frame", variant="secondary"
                    )

        with gr.Column(visible=False) as gen_animation_col:
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="Prompt")
                    gen_animation_button = gr.Button(
                        value="Generate Animation", variant="primary"
                    )

                    with gr.Accordion("Advanced options", open=False):
                        n_prompt = gr.Textbox(
                            label="Negative Prompt (optional)", value=""
                        )

                        if on_huggingspace:
                            video_length = gr.Slider(
                                label="Video length", minimum=8, maximum=16, step=1
                            )
                        else:
                            video_length = gr.Number(
                                label="Video length", value=8, precision=0
                            )

                        seed = gr.Slider(
                            label="Seed",
                            info="-1 for random seed on each run. Otherwise, the seed will be fixed.",
                            minimum=-1,
                            maximum=65536,
                            value=0,
                            step=1,
                        )

                        motion_field_strength_x = gr.Slider(
                            label="Global Translation $\\delta_{x}$",
                            minimum=-20,
                            maximum=20,
                            value=12,
                            step=1,
                        )

                        motion_field_strength_y = gr.Slider(
                            label="Global Translation $\\delta_{y}$",
                            minimum=-20,
                            maximum=20,
                            value=12,
                            step=1,
                        )

                        t0 = gr.Slider(
                            label="Timestep t0",
                            minimum=0,
                            maximum=47,
                            value=44,
                            step=1,
                            info="Perform DDPM steps from t0 to t1. The larger the gap between t0 and t1, the more variance between the frames. Ensure t0 < t1 ",
                        )

                        t1 = gr.Slider(
                            label="Timestep t1",
                            minimum=1,
                            info="Perform DDPM steps from t0 to t1. The larger the gap between t0 and t1, the more variance between the frames. Ensure t0 < t1",
                            maximum=48,
                            value=47,
                            step=1,
                        )

                        chunk_size = gr.Slider(
                            label="Chunk size",
                            minimum=2,
                            maximum=16,
                            value=8,
                            step=1,
                            visible=not on_huggingspace,
                            info="Number of frames processed at once. Reduce for lower memory usage.",
                        )
                        merging_ratio = gr.Slider(
                            label="Merging ratio",
                            minimum=0.0,
                            maximum=0.9,
                            step=0.1,
                            value=0.0,
                            visible=not on_huggingspace,
                            info="Ratio of how many tokens are merged. The higher the more compression (less memory and faster inference).",
                        )

                with gr.Column():
                    result = gr.Image(label="Generated Video")

        inputs = [
            prompt,
            seed,
            input_video_path,
            lora_scale,
            # prompt, 
            # # model_link,
            # # is_safetensor,
            # motion_field_strength_x,
            # motion_field_strength_y,
            # t0,
            # t1,
            # n_prompt,
            # chunk_size,
            # video_length,
            # merging_ratio,
            # seed,
        ]

        # gr.Examples(examples=examples,
        #             inputs=inputs,
        #             outputs=result,
        #             fn=None,
        #             run_on_click=False,
        #             cache_examples=on_huggingspace,
        # )

        frame_inputs = [
            frames_prompt, num_imgs, input_video_path]

        def submit_select():
            show = True
            if initial_frame is not None:  # More to next step
                return {
                    frame_selection_col: gr.update(visible=not show),
                    gen_animation_col: gr.update(visible=show),
                }

            return {
                frame_selection_col: gr.update(visible=show),
                gen_animation_col: gr.update(visible=not show),
            }

        gen_frames_button.click(
            generate_initial_frames,
            inputs=frame_inputs,
            outputs=[initial_frames],
        )
        select_frame_button.click(
            submit_select, inputs=None, outputs=[frame_selection_col, gen_animation_col]
        )

        gen_animation_button.click(
            fn=generate_video_frames,
            inputs=inputs,
            outputs=result,
        )

    return demo
