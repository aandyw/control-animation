python ./train_textual_inversion_flax.py \
--pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
--revision "flax" \
--train_data_dir "../aardman/imgs" \
--learnable_property="style" \
--placeholder_token="<aardman>" --initializer_token="clay" \
--resolution=512 \
--train_batch_size=4 \
--max_train_steps=3000 \
--learning_rate=5.0e-04 --scale_lr \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--push_to_hub \
--output_dir="textual_inversion_aardman"