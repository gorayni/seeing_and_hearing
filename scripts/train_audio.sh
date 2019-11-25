#!/usr/bin/bash

export gpu_number="$1"

if test "$#" -eq 2; then
   export classification_type="$2"
else
   export classification_type="verb"
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
cat $conf_fname | jq '.train=true' > $tmpfile
mv $tmpfile $conf_fname

lr=$(cat $conf_fname | jq .lr)
arc=$(cat $conf_fname | jq .arc | tr -d '"')
log_fname="$arc"_"$classification_type"_`timestamp`.log 

msg1="Training audio for $classification_type using $arc with learning rate $lr started on GPU $gpu_number"
msg2="Training audio for $classification_type using $arc with learning rate $lr finished on GPU $gpu_number"

python3 audio_main.py "$conf_fname" </dev/null 2>&1 | tee "$log_fname" &
msg_me "$!" "$msg1" "$msg2"

