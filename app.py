import gradio as gr

from text_to_animation.model import ControlAnimationModel
from webui.app_pose import create_demo as create_demo_pose
from webui.app_text_to_video import create_demo as create_demo_text_to_video
from webui.app_control_animation import create_demo as create_demo_animation
import argparse
import os
import jax.numpy as jnp

huggingspace_name = os.environ.get("SPACE_AUTHOR_NAME")
on_huggingspace = huggingspace_name if huggingspace_name is not None else False

model = ControlAnimationModel(device="cuda", dtype=jnp.float16)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--public_access",
    action="store_true",
    help="if enabled, the app can be access from a public url",
    default=False,
)
args = parser.parse_args()


title = """
<div style="text-align: center; max-width: 1200px; margin: 20px auto;">
<h1 style="font-weight: 900; font-size: 3rem; margin: 0rem">Control Animation</h1>
"""

description = """
<div style="text-align: center; max-width: 1200px; margin: 20px auto;">
<h2 style="font-weight: 450; font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem">
Our code uses <a href="https://www.humphreyshi.com/home">Text2Video-Zero</a> and the <a href="https://github.com/huggingface/diffusers">Diffusers</a> library as inspiration.
</h2>
</div>
"""

notice = """
<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings.
<br/>
<a href="https://huggingface.co/spaces/Pie31415/control-animation?duplicate=true">
<img style="margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>
</p>
"""

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    if on_huggingspace:
        gr.HTML(notice)

    with gr.Tab("Control Animation"):
        create_demo_animation(model)

if on_huggingspace:
    demo.queue(max_size=20)
    demo.launch(debug=True)
else:
    _, _, link = demo.queue(api_open=False).launch(
        file_directories=["temporal"], share=args.public_access, debug=True
    )
    print(link)
