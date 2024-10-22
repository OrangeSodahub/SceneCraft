#!/usr/bin/env bash

# scannet
scannet_data_root="./data/scannet"
scannet_output_path="./data/scannet/semantic_images"
scannet_jsonl_path="./data/scannet"
# scannetpp
scannetpp_data_root="./data/scannetpp_processed"
scannetpp_output_path="./data/scannetpp_processed/semantic_images"
scannetpp_jsonl_path="./data/scannetpp_processed"
# hypersim
hypersim_data_root="./data/hypersim"
hypersim_output_path="./data/hypersim/semantic_images"
hypersim_jsonl_path="./data/hypersim"


DATASET=$1
LIMIT=$2

if [ -z "$DATASET" ]; then
    echo "Error: DATASET is missing, should be one of [scannet, scannetpp]"
    exit 1
elif [ "$DATASET" == "scannet" ]; then
    DATA_ROOT="$scannet_data_root"
    OUTPUT_PATH="$scannet_output_path"
    JSONL_PATH="$scannet_jsonl_path"
elif [ "$DATASET" == "scannetpp" ]; then
    DATA_ROOT="$scannetpp_data_root"
    OUTPUT_PATH="$scannetpp_output_path"
    JSONL_PATH="$scannetpp_jsonl_path"
elif [ "$DATASET" == "hypersim" ]; then
    DATA_ROOT="$hypersim_data_root"
    OUTPUT_PATH="$hypersim_output_path"
    JSONL_PATH="$hypersim_jsonl_path"
else
    echo "Error: Unknown dataset type $DATASET, should be one of [scannet, scannetpp]"
    exit 1
fi

if [ -z "$LIMIT" ]; then
    LIMIT=-1
    echo "LIMIT is set to -1"
fi


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python $(dirname "$0")/../scenecraft/renderer.py \
            --dataset="$DATASET" --data-root="$DATA_ROOT" --output-path="$OUTPUT_PATH" --jsonl-path="$JSONL_PATH" --limit "$LIMIT" --generate-json ${@:3}
