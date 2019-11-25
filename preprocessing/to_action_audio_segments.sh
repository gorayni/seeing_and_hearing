#!/bin/bash

export action_audio_segments="$1"
export data_split="$2"

cat $action_audio_segments | \
                      sort | \
                tr ',' ' ' | \
                parallel -P50  --colsep ' ' './action_segment_to_wav.sh {1} {2} {3} {4} {5} {6} $data_split'
