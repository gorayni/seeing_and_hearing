#!/bin/bash

export gpu_number="$1"
export modality="$2"
if test "$#" -eq 3; then
   export split_name="$3"
else
   export split_name="test"
fi

conda activate ~/anaconda3/envs/epic_torch

export CUDA_VISIBLE_DEVICES=$gpu_number

export root_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. >/dev/null && pwd )"
export PYTHONPATH="$root_dir/IO:$root_dir/epic-lib"

cd "$root_dir"

conf_fname=conf/tsn_"$modality"_baradel_args_"$gpu_number".json

export splits=$(cat $conf_fname | jq .splits | tr -d '"')
export save_scores=$(cat $conf_fname | jq .save_scores | tr -d '"')
export num_iterations="$(python - <<END
import pickle
from math import ceil
from os import environ

splits_path = environ['splits']
split_name = environ['split_name']
splits = pickle.load(open(splits_path, 'rb'))
print(ceil(len(splits[split_name])/100))
END
)"

if [ "$split_name" != "test" ]; then
    save_scores=`echo $save_scores | sed 's/.npz$/_'$split_name'.npz/g'`
fi

for i in `seq 1 $num_iterations`;
do
    scores_file=`echo $save_scores | sed 's/\.npz/_part-'$i'\.npz/g'`
    [ -f $scores_file ] && continue

    python tsn_test_main.py "$conf_fname" -split_name "$split_name" -part $i

    sleep 15s
done

python - <<END
import numpy as np
import os


num_iterations = int(os.environ['num_iterations'])
save_scores = os.environ['save_scores']

segment_indices, scores, labels = [], [], []
for i in range(1, num_iterations+1):
    npz_filepath = save_scores.replace('.npz', '_part-{}.npz'.format(i))
    results = np.load(npz_filepath)

    segment_indices.extend(results['segment_indices'])
    scores.extend(results['scores'])
    labels.extend(results['labels'])
np.savez(save_scores, segment_indices=segment_indices, scores=scores, labels=labels)
END
