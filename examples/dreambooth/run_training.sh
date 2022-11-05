#!/bin/bash

export MODEL_NAME='CompVis/stable-diffusion-v1-4'
export INSTANCE_DIR='s3://sagemaker-eu-west-1-485824217930/user_data'
export INSTANCE_PROMPT='a photo of sks man'
export OUTPUT_DIR='s3://sagemaker-eu-west-1-485824217930/user_model'
export HF_TOKEN='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'

accelerate launch train_dreambooth.py \
    --pretrained_model_name_or_path $MODEL_NAME \
    --instance_data_dir $INSTANCE_DIR \
    --output_dir $OUTPUT_DIR \
    --hf_auth_token $HF_TOKEN \
    --instance_prompt "a photo of sks man" \
    --resolution 512 \
    --train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-6 \
    --lr_scheduler "constant" \
    --lr_warmup_steps 0 \
    --max_train_steps 800