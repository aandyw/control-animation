import gradio as gr
from text_to_animation.model import ControlAnimationModel
import os
from utils.hf_utils import get_model_list

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


images = []  # str path of generated images
initial_frame = None
animation_model = None


def generate_initial_frames(
    frames_prompt,
    model_link,
    is_safetensor,
    frames_n_prompt,
    width,
    height,
    cfg_scale,
    seed,
):
    global images

    if not model_link:
        model_link = "dreamlike-art/dreamlike-photoreal-2.0"

    images = animation_model.generate_initial_frames(
        frames_prompt,
        model_link,
        is_safetensor,
        frames_n_prompt,
        width,
        height,
        cfg_scale,
        seed,
    )

    return images


def select_initial_frame(evt: gr.SelectData):
    global initial_frame

    if evt.index < len(images):
        initial_frame = images[evt.index]
        print(initial_frame)


def create_demo(model: ControlAnimationModel):
    global animation_model
    animation_model = model

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
                    model_link = gr.Textbox(
                        label="Model Link",
                        placeholder="dreamlike-art/dreamlike-photoreal-2.0",
                        info="Give the hugging face model name or URL link to safetensor.",
                    )
                    is_safetensor = gr.Checkbox(label="Safetensors")
                    gen_frames_button = gr.Button(
                        value="Generate Initial Frames", variant="primary"
                    )

            with gr.Row():
                with gr.Column(scale=2):
                    width = gr.Slider(32, 2048, value=512, label="Width")
                    height = gr.Slider(32, 2048, value=512, label="Height")
                    cfg_scale = gr.Slider(1, 20, value=7.0, step=0.1, label="CFG scale")
                    seed = gr.Slider(
                        label="Seed",
                        info="-1 for random seed on each run. Otherwise, the seed will be fixed.",
                        minimum=-1,
                        maximum=65536,
                        value=0,
                        step=1,
                    )

                with gr.Column(scale=3):
                    initial_frames = gr.Gallery(
                        label="Initial Frames", show_label=False
                    ).style(
                        columns=[2], rows=[2], object_fit="scale-down", height="auto"
                    )
                    initial_frames.select(select_initial_frame)
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
                    result = gr.Video(label="Generated Video")

        inputs = [
            prompt,
            model_link,
            is_safetensor,
            motion_field_strength_x,
            motion_field_strength_y,
            t0,
            t1,
            n_prompt,
            chunk_size,
            video_length,
            merging_ratio,
            seed,
        ]

        # gr.Examples(examples=examples,
        #             inputs=inputs,
        #             outputs=result,
        #             fn=None,
        #             run_on_click=False,
        #             cache_examples=on_huggingspace,
        # )

        frame_inputs = [
            frames_prompt,
            model_link,
            is_safetensor,
            frames_n_prompt,
            width,
            height,
            cfg_scale,
            seed,
        ]

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
            outputs=initial_frames,
        )
        select_frame_button.click(
            submit_select, inputs=None, outputs=[frame_selection_col, gen_animation_col]
        )

        gen_animation_button.click(
            fn=model.process_text2video,
            inputs=inputs,
            outputs=result,
        )

    return demo
