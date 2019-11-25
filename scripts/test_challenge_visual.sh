#!/bin/bash

export gpu_number="$1"
export modality="$2"
export classification_type="$3"

export root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null && pwd )"

# Setting Nvidia GPU and Anancoda virtual environment path
conda activate ~/anaconda3/envs/epic_torch
export CUDA_VISIBLE_DEVICES=$gpu_number

export PYTHONPATH="$root_dir/IO:$root_dir/epic-lib"

cd "$root_dir"

conf_fname=conf/tsn_"$modality"_"$classification_type"_args_"$gpu_number".json

export save_scores=$(cat $conf_fname | jq .save_scores | tr -d '"')
export checkpoint=$(cat $conf_fname | jq .checkpoint | tr -d '"')
export class_type=$(cat $conf_fname | jq .class_type | tr -d '"')
export arch=$(cat $conf_fname | jq .arch | tr -d '"')
export lr=$(cat $conf_fname | jq .lr | tr -d '"')
export weights_path=$(cat $conf_fname | jq .weights | tr -d '"')
export epoch="$(python - <<END
from os import environ
import torch

weights_path = environ['weights_path']
checkpoint = torch.load(weights_path)
print(checkpoint['epoch'])
END
)"

test_sets='s1 s2'
for test_set in $test_sets
do
    export final_scores_file="$checkpoint"/tsn_"$class_type"_"$modality"_testset_"$test_set"_"$arch"_lr_"$lr"_model_`printf "%03d\n" $epoch`.npz
    num_tests_samples=`tail -n+2 $root_dir/annotations/EPIC_test_"$test_set"_timestamps.csv | wc -l`
    export num_iterations="$(python -c "from math import ceil; print(ceil($num_tests_samples/100.))")"

    for i in `seq 1 $num_iterations`;
    do
        scores_file=`echo $final_scores_file | sed 's/\.npz/_part-'$i'\.npz/g'`
        [ -f $scores_file ] && continue

        python test_epic_challenge_tsn.py "$conf_fname" "$test_set" -part $i

        sleep 15s
    done

python - <<END
import numpy as np
import os


num_iterations = int(os.environ['num_iterations'])
save_scores = os.environ['final_scores_file']

segment_indices, scores, labels = [], [], []
for i in range(1, num_iterations+1):
    npz_filepath = save_scores.replace('.npz', '_part-{}.npz'.format(i))
    results = np.load(npz_filepath)

    segment_indices.extend(results['segment_indices'])
    scores.extend(results['scores'])
np.savez(save_scores, segment_indices=segment_indices, scores=scores)
END

done
