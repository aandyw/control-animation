import gradio as gr
from text_to_animation.model import ControlAnimationModel
import os
from utils.hf_utils import get_model_list

huggingspace_name = os.environ.get("SPACE_AUTHOR_NAME")
on_huggingspace = huggingspace_name if huggingspace_name is not None else False

examples = []
# examples = [
#     ["an astronaut waving the arm on the moon"],
#     ["a sloth surfing on a wakeboard"],
#     ["an astronaut walking on a street"],
#     ["a cute cat walking on grass"],
#     ["a horse is galloping on a street"],
#     ["an astronaut is skiing down the hill"],
#     ["a gorilla walking alone down the street"],
#     ["a gorilla dancing on times square"],
#     ["A panda dancing dancing like crazy on Times Square"],
# ]

def on_video_path_update(evt: gr.EventData):
    return f"Selection: **{evt._data}**"

def pose_gallery_callback(evt: gr.SelectData):
    return f"Motion {evt.index+1}"


def get_frame_index(evt: gr.SelectData):
    return evt.index


def create_demo(model: ControlAnimationModel):
    with gr.Blocks() as demo:
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    # TODO: update so that model_link is customizable
                    model_link = gr.Dropdown(
                        label="Model Link",
                        choices=["runwayml/stable-diffusion-v1-5"],
                        value="runwayml/stable-diffusion-v1-5",
                    )
                    prompt = gr.Textbox(
                        placeholder="Prompt",
                        show_label=False,
                        lines=2,
                        info="Give a prompt for an animation you would like to generate. The prompt will be used to create the first initial frame and then the animation.",
                    )
                    negative_prompt = gr.Textbox(
                        placeholder="Negative Prompt (optional)",
                        show_label=False,
                        lines=2,
                    )

                    gen_frames_button = gr.Button(
                        value="Generate Initial Frames", variant="primary"
                    )

                    with gr.Accordion("Advanced options", open=False):
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
                    gallery_pose_sequence = gr.Gallery(
                        label="Pose Sequence",
                        value=[
                            ("__assets__/dance1.gif", "Motion 1"),
                            ("__assets__/dance2.gif", "Motion 2"),
                            ("__assets__/dance3.gif", "Motion 3"),
                            ("__assets__/dance4.gif", "Motion 4"),
                            ("__assets__/dance5.gif", "Motion 5"),
                            ("__assets__/walk_01.gif", "Motion 6"),
                        ],
                    ).style(columns=3)
                    input_video_path = gr.Textbox(
                        label="Pose Sequence", visible=False, value="Motion 1"
                    )
                    pose_sequence_selector = gr.Markdown("Pose Sequence: **Motion 1**")

            with gr.Row():
                with gr.Column(visible=True) as frame_selection_view:
                    initial_frames = gr.Gallery(
                        label="Initial Frames", show_label=False
                    ).style(columns=4, rows=1, object_fit="contain", preview=True)

                    gr.Markdown("Select an initial frame to start your animation with.")

                    gen_animation_button = gr.Button(
                        value="Select Initial Frame & Generate Animation",
                        variant="secondary",
                    )

                with gr.Column(visible=False) as animation_view:
                    result = gr.Image(label="Generated Video")

        with gr.Box(visible=False):
            initial_frame_index = gr.Number(
                label="Selected Initial Frame Index", value=-1, precision=0
            )

        input_video_path.change(on_video_path_update, None, pose_sequence_selector)
        gallery_pose_sequence.select(pose_gallery_callback, None, input_video_path)
        initial_frames.select(fn=get_frame_index, outputs=initial_frame_index)

        frame_inputs = [
            prompt,
            input_video_path,
            negative_prompt,
        ]

        animation_inputs = [
            prompt,
            initial_frame_index,
            input_video_path,
            model_link,
            motion_field_strength_x,
            motion_field_strength_y,
            t0,
            t1,
            negative_prompt,
            chunk_size,
            video_length,
            merging_ratio,
            seed,
        ]

        def submit_select(initial_frame_index: int):
            if initial_frame_index != -1:  # More to next step
                return {
                    frame_selection_view: gr.update(visible=False),
                    animation_view: gr.update(visible=True),
                }

            return {
                frame_selection_view: gr.update(visible=True),
                animation_view: gr.update(visible=False),
            }

        gen_frames_button.click(
            fn=model.generate_initial_frames,
            inputs=frame_inputs,
            outputs=initial_frames,
        )

        gen_animation_button.click(
            fn=submit_select,
            inputs=initial_frame_index,
            outputs=[frame_selection_view, animation_view],
        ).then(
            fn=model.generate_animation,
            inputs=animation_inputs,
            outputs=result,
        )

        # gr.Examples(examples=examples,
        #             inputs=inputs,
        #             outputs=result,
        #             fn=None,
        #             run_on_click=False,
        #             cache_examples=on_huggingspace,
        # )

        gr.Examples(examples=examples,
                    inputs=animation_inputs,
                    outputs=result,
                    fn=model.generate_animation,
                    cache_examples=on_huggingspace,
                    run_on_click=True,
                    )

    return demo
