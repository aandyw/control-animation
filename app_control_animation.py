import gradio as gr
from pipelines.model import ControlAnimationModel
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


# TODO: CREATE A GRADIO INTERFACE THAT ALLOWS YOU TO GENERATE 9 DIFFERENT FRAMES AND PICK ONE AS A STARTING POINT.

def create_demo(model):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown("## Control Animation")

        with gr.Row():
          with gr.Column():
            model_link = gr.Textbox(label='Model Link')
            is_safetensor = gr.Checkbox(label='Safetensors')
            prompt = gr.Textbox(label='Prompt')
            run_button = gr.Button(label='Run')
            
            with gr.Accordion("Advanced options", open=False):
              n_prompt = gr.Textbox(label="Negative Prompt (optional)", value='')

              if on_huggingspace:
                video_length = gr.Slider(label="Video length", minimum=8, maximum=16, step=1)
              else:
                video_length = gr.Number(label="Video length", value=8, precision=0)

              seed = gr.Slider(label='Seed',
                              info="-1 for random seed on each run. Otherwise, the seed will be fixed.",
                              minimum=-1,
                              maximum=65536,
                              value=0,
                              step=1)

              motion_field_strength_x = gr.Slider(
                  label='Global Translation $\\delta_{x}$', minimum=-20, maximum=20,
                  value=12,
                  step=1)
              
              motion_field_strength_y = gr.Slider(
                  label='Global Translation $\\delta_{y}$', minimum=-20, maximum=20,
                  value=12,
                  step=1)

              t0 = gr.Slider(label="Timestep t0", minimum=0,
                              maximum=47, value=44, step=1,
                              info="Perform DDPM steps from t0 to t1. The larger the gap between t0 and t1, the more variance between the frames. Ensure t0 < t1 ",
                              )
              
              t1 = gr.Slider(label="Timestep t1", minimum=1,
                              info="Perform DDPM steps from t0 to t1. The larger the gap between t0 and t1, the more variance between the frames. Ensure t0 < t1",
                              maximum=48, value=47, step=1)
              
              chunk_size = gr.Slider(
                  label="Chunk size", minimum=2, maximum=16, value=8, step=1, visible=not on_huggingspace,
                  info="Number of frames processed at once. Reduce for lower memory usage."
              )
              merging_ratio = gr.Slider(
                  label="Merging ratio", minimum=0.0, maximum=0.9, step=0.1, value=0.0, visible=not on_huggingspace,
                  info="Ratio of how many tokens are merged. The higher the more compression (less memory and faster inference)."
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

          run_button.click(fn=None,
                           inputs=inputs,
                           outputs=result,)
        return demo