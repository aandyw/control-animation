import jax
import jax.numpy as jnp
from transformers import CLIPTokenizer, CLIPFeatureExtractor, FlaxCLIPTextModel
from diffusers import FlaxDDIMScheduler, FlaxAutoencoderKL
import gradio_utils
import utils
from flax.training.common_utils import shard
from flax import jax_utils
import flax

from custom_flaxunet2D.unet_2d_condition_flax import FlaxUNet2DConditionModel, FlaxLoRAUNet2DConditionModel
from custom_flaxunet2D.controlnet_flax import FlaxControlNetModel

unet, params = FlaxLoRAUNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", revision="fp16", subfolder="unet", from_pt=True)

latent_model_input = jnp.zeros((1, 4, 64, 64))
timestep = jnp.broadcast_to(0, latent_model_input.shape[0])
encoder_hidden_states= jnp.zeros((1, 77, 768))
rng = jax.random.PRNGKey(0)
random_params = unet.init(rng, latent_model_input, timestep, encoder_hidden_states) # Initialization call

def tree_copy(tree1, tree2):
    #copies from tree2 to tree1
    # print(tree1)
    for k in tree1.keys():
        if k in ["bias", "kernel", "scale"]:
            tree2[k] = tree1[k]
        else:
            tree_copy(tree1[k], tree2[k])
    return tree2

def freeze_non_lora(nested_dict):
    """recursively check if '_lora' is in the name'"""
    r = {}
    for k,v in nested_dict.items():
        if '_lora' in k:
            r[k] = "lora"
        else:
            if isinstance(v, dict):
                r[k] = freeze_non_lora(v)
            else:
                r[k] = "freeze"
    return r

lora_unet_params = tree_copy(params, flax.core.frozen_dict.unfreeze(random_params["params"]))
mask = freeze_non_lora(lora_unet_params)

