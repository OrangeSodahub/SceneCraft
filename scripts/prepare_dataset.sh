#!/usr/bin/env bash

# scannet
scannet_raw_data_root=""
scannet_data_root="./data/scannet"
scannet_output_path="./data/scannet/semantic_images"
# scannetpp
scannetpp_raw_data_root="./data/scannetpp"
scannetpp_data_root="./data/scannetpp_processed"
scannetpp_output_path="./data/scannetpp_processed/semantic_images"
# hypersim
hypersim_raw_data_root=""
hypersim_data_root="./data/hypersim"
hypersim_output_path="./data/hypersim/semantic_images"
# custom
EXP="exp"
custom_data_root="./exp"
custom_data_output_path="./exp"

DATASET=$1
LIMIT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

if [ -z "$DATASET" ]; then
    echo "Error: DATASET is missing, should be one of [scannet, scannetpp, hypersim]"
    exit 1
elif [ "$DATASET" == "scannet" ]; then
    RAW_DATA_ROOT="$scannet_raw_data_root"
    DATA_ROOT="$scannet_data_root"
    OUTPUT_PATH="$scannet_output_path"
elif [ "$DATASET" == "scannetpp" ]; then
    RAW_DATA_ROOT="$scannetpp_raw_data_root"
    DATA_ROOT="$scannetpp_data_root"
    OUTPUT_PATH="$scannetpp_output_path"
elif [ "$DATASET" == "hypersim" ]; then
    RAW_DATA_ROOT="$hypersim_raw_data_root"
    DATA_ROOT="$hypersim_data_root"
    OUTPUT_PATH="$hypersim_output_path"
elif [[ "$DATASET" == *$exp* ]]; then
    RAW_DATA_ROOT=""
    DATA_ROOT="$custom_data_root"
    OUTPUT_PATH="$custom_data_root/semantic_images"
else
    echo "Error: Unknown dataset type $DATASET, should be one of [scannet, scannetpp, hypersim]"
    exit 1
fi

if [ -z "$LIMIT" ]; then
    echo "Error: LIMIT parameter is missing."
    exit 1
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --nproc_per_node="$GPUS" \
    --master_port="$PORT" \
    $(dirname "$0")/../dreamscene/renderer.py \
    --dataset="$DATASET" \
    --raw-data-root=$RAW_DATA_ROOT \
    --data-root=$DATA_ROOT \
    --output-path=$OUTPUT_PATH \
    --limit "$LIMIT" \
    --save-image \
    --launcher pytorch ${@:4}
