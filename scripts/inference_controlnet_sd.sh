#!/usr/bin/env bash

DATASET=$1

if [ -z "$DATASET" ] || { [ "$DATASET" != "scannet" ] && [ "$DATASET" != "scannetpp" ]; }; then
    echo "Error: DATASET is missing, should be one of [scannet, scannetpp]"
    exit 1
fi

# Must be the same as the 'pretrained_model_name_or_path' in 'train_controlnet_sd.sh'
BASE_MODEL_PATH="runwayml/stable-diffusion-v1-5"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/../dreamscene/finetune/inference_controlnet_sd.py \
    --dataset="$DATASET" \
    --base_model_path="$BASE_MODEL_PATH" ${@:2}