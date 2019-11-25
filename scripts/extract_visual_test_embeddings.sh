#!/bin/bash

export gpu_number="$1"
export modality="$2"
export classification_type="$3"

export root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null && pwd )"

conda activate ~/anaconda3/envs/epic_torch

export CUDA_VISIBLE_DEVICES=$gpu_number

export PYTHONPATH="$root_dir/IO:$root_dir/epic-lib"

cd "$root_dir"


test_sets='s1 s2'
for test_set in $test_sets
do
    export num_tests_samples=`tail -n+2 $root_dir/annotations/EPIC_test_"$test_set"_timestamps.csv | wc -l`
    export num_iterations="$(python -c "from math import ceil; print(ceil($num_tests_samples/100.))")"

    conf_fname=conf/tsn_"$modality"_"$classification_type"_args_"$gpu_number".json
    embeddings=embeddings_"$classification_type"_"$modality"_"$test_set"_testing.pkl

    for i in `seq 1 $num_iterations`;
    do
        python extract_tsn_features.py "$conf_fname" "$embeddings" -test_set "$test_set" -part $i
        sleep 15s
    done
done
