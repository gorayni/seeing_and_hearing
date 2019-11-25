#!/bin/bash

export gpu_number="$1"

if test "$#" -lt 2; then
   export classification_type="verb"
else
   export classification_type="$2"
fi

export root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null && pwd )"

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

embeddings=embeddings_"$classification_type"_training.pkl

for i in `seq 1 $num_iterations`;
do
    
    sleep 15s
done 
python3 extract_audio_features.py "$conf_fname" "$embeddings"
