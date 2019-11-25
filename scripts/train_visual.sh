#!/bin/bash

export gpu_number="$1"

if test "$#" -lt 2; then
   export modality="flow"
else
   export modality="$2"
fi

if test "$#" -eq 3; then
   export classification_type="$3"
else
   export classification_type="verb"
fi

export root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null && pwd )"

# Setting Nvidia GPU and Anancoda virtual environment path
conda activate ~/anaconda3/envs/epic_torch
export CUDA_VISIBLE_DEVICES=$gpu_number

export PYTHONPATH="$root_dir/IO:$root_dir/epic-lib"

source "$root_dir"/scripts/commons.sh

cd "$root_dir"

conf_fname=conf/tsn_"$modality"_"$classification_type"_args_"$gpu_number".json

lr=$(cat $conf_fname | jq .lr)
arch=$(cat $conf_fname | jq .arch | tr -d '"')
log_fname=tsn_"$arch"_"$modality"_"$classification_type"_lr:"$lr"_`timestamp`.log

msg1="Training TSN $modality for $classification_type with learning rate $lr started on GPU $gpu_number"
msg2="Training TSN $modality for $classification_type with learning rate $lr finished on GPU $gpu_number"

python3 tsn_main.py "$conf_fname" </dev/null 2>&1 | tee "$log_fname" &
msg_me "$!" "$msg1" "$msg2"
