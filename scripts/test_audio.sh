#!/usr/bin/bash

export gpu_number="$1"

if test "$#" -lt 2; then
   export classification_type="verb"
else
   export classification_type="$2"
fi

if test "$#" -eq 3; then
   export split_name="$3"
else
   export split_name="test"
fi

export root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null && pwd )"

# Setting Nvidia GPU and Anancoda virtual environment path
conda activate ~/anaconda3/envs/epic_torch

export CUDA_VISIBLE_DEVICES=$gpu_number
export PYTHONPATH="$root_dir" 

source "$root_dir"/scripts/commons.sh

cd "$root_dir"

if [ "$gpu_number" -eq "0" ]; then
    conf_fname=conf/audio_"$classification_type"_args.json
else
    conf_fname=conf/audio_"$classification_type"_args_"$gpu_number".json
fi

tmpfile=$(mktemp /tmp/conf.json.XXXXXX)
cat $conf_fname | jq '.train=false' > $tmpfile
mv $tmpfile $conf_fname

lr=$(cat $conf_fname | jq .lr)
arc=$(cat $conf_fname | jq .arc | tr -d '"')

msg1="Testing audio for $classification_type using $arc with learning rate $lr started on GPU $gpu_number"
msg2="Testing audio for $classification_type using $arc with learning rate $lr finished on GPU $gpu_number"

python3 audio_main.py "$conf_fname" -split_name "$split_name" </dev/null 2>&1 &
msg_me "$!" "$msg1" "$msg2"
