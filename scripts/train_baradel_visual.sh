#!/bin/bash

export gpu_number="$1"
export modality="$2"

export classification_type="verb"
export root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null && pwd )"

# Setting Nvidia GPU and Anancoda virtual environment path
conda activate ~/anaconda3/envs/epic_torch
export CUDA_VISIBLE_DEVICES=`awk -v gpu_number="$gpu_number" 'BEGIN {print (gpu_number+1)%2}'`

export PYTHONPATH="$root_dir/IO:$root_dir/epic-lib"

source "$root_dir"/scripts/commons.sh

cd "$root_dir"

conf_fname=conf/tsn_"$modality"_baradel_args_"$gpu_number".json

lr=$(cat $conf_fname | jq .lr)
arch=$(cat $conf_fname | jq .arch | tr -d '"')
log_fname=tsn_baradel_"$arch"_"$modality"_"$classification_type"_lr:"$lr"_`timestamp`.log

msg1="Training TSN $modality for $classification_type with learning rate $lr started on GPU $gpu_number"
msg2="Training TSN $modality for $classification_type with learning rate $lr finished on GPU $gpu_number"

python3 tsn_main.py "$conf_fname" </dev/null 2>&1 | tee "$log_fname" &
msg_me "$!" "$msg1" "$msg2"

