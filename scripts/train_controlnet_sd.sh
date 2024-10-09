#!/usr/bin/env bash

DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Error: DATASET is missing, should be one of [scannet, scannetpp]"
    exit 1
fi

MODEL_NAME="stabilityai/stable-diffusion-2-1"
OUTPUT_DIR="./outputs/finetune/controlnet/$DATASET"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
accelerate launch \
 --mixed_precision="fp16" --multi_gpu \
 $(dirname "$0")/../scenecraft/finetune/train_controlnet_sd.py \
 --pretrained_model_name_or_path=$MODEL_NAME \
 --dataset="$DATASET" \
 --output_dir="$OUTPUT_DIR" \
 --controlnet_conditioning_scale=3.5 \
 --train_batch_size=16 \
 --num_train_epochs=18 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=100 \
 --validation_steps=100 \
 --seed=1337 ${@:2}