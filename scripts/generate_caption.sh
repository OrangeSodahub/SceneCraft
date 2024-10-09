#!/usr/bin/env bash

# scannet
scannet_data_root="./data/scannet"
scannet_output_path="./data/scannet/captions"
# scannetpp
scannetpp_data_root="./data/scannetpp_processed"
scannetpp_output_path="./data/scannetpp_processed/captions"
# hypersim
hypersim_data_root="./data/hypersim"
hypersim_output_path="./data/hypersim/captions"

DATASET=$1

if [ -z "$DATASET" ]; then
    echo "Error: DATASET is missing, should be one of [scannet, scannetpp]"
    exit 1
elif [ "$DATASET" == "scannet" ]; then
    DATA_ROOT="$scannet_data_root"
    OUTPUT_PATH="$scannet_output_path"
    JSONL_PATH="$scannet_data_root"
elif [ "$DATASET" == "scannetpp" ]; then
    DATA_ROOT="$scannetpp_data_root"
    OUTPUT_PATH="$scannetpp_output_path"
    JSONL_PATH="$scannetpp_data_root"
elif [ "$DATASET" == "hypersim" ]; then
    DATA_ROOT="$hypersim_data_root"
    OUTPUT_PATH="$hypersim_output_path"
    JSONL_PATH="$hypersim_data_root"
else
    echo "Error: Unknown dataset type $DATASET, should be one of [scannet, scannetpp]"
    exit 1
fi


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/../dreamscene/finetune/prepare_dataset.py \
            --dataset="$DATASET" --data-root="$DATA_ROOT" --output-path="$OUTPUT_PATH" --jsonl-path="$JSONL_PATH" --generate-caption ${@:3}
