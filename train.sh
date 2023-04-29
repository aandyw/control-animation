python train_lora_flax.py \
--pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
--revision fp16 \
--hub_model_id gigant/lora-t2vz-sd15 \
--mixed_precision fp16 \
--revision flax\