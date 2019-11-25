#!/bin/bash

export gpu_number="$1"
export modality="$2"
export classification_type="$3"

export root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null && pwd )"

conda activate ~/anaconda3/envs/epic_torch

export CUDA_VISIBLE_DEVICES=$gpu_number

export PYTHONPATH="$root_dir/IO:$root_dir/epic-lib"

cd "$root_dir"

export num_samples=`tail -n+2 annotations/EPIC_train_action_labels.csv | wc -l`
export num_iterations="$(python -c "from math import ceil; print(ceil($num_samples/100.))")"

conf_fname=conf/tsn_"$modality"_"$classification_type"_args_"$gpu_number".json
embeddings=embeddings_"$classification_type"_training.pkl

for i in `seq 1 $num_iterations`;
do
    python extract_tsn_features.py "$conf_fname" "$embeddings" -part $i
    sleep 15s
done 
