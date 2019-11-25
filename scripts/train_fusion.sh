#!/bin/bash

export gpu_number="$1"

if test "$#" -lt 2; then
   export modalities="flow"
else
   export modalities="$2"
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

conf_fname=conf/fusion_"$classification_type"_"$modalities"_args_"$gpu_number".json

tmpfile=$(mktemp /tmp/conf.json.XXXXXX)
cat $conf_fname | jq '.train=true' > $tmpfile
mv $tmpfile $conf_fname

lr=$(cat $conf_fname | jq .lr)
log_fname=fusion_"$classification_type"_"$modalities"_lr:"$lr"_`timestamp`.log

msg1="Training Fusion $modalities for $classification_type with learning rate $lr started on GPU $gpu_number"
msg2="Training Fusion $modalities for $classification_type with learning rate $lr finished on GPU $gpu_number"

python3 fusion_main.py "$conf_fname" </dev/null 2>&1 | tee "$log_fname" &
msg_me "$!" "$msg1" "$msg2"

