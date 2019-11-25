#!/bin/bash

export test_set=$1; #s1 or s2
export frames_dir="EPIC_KITCHENS_2018/frames_rgb_flow"
export test_labels_path="annotations/EPIC_test_"$test_set"_timestamps.pkl"
export test_labels_csv_path="annotations/EPIC_test_"$test_set"_timestamps.csv"

for participant_id in `tail -n +2 $test_labels_csv_path | cut -d',' -f3 | sort | uniq`; do
    find $frames_dir -mindepth 4 \
                     -maxdepth 4 \
                     -type d \
                     -name "$participant_id" | \
                     tr '/' ' ' | \
                     xargs -n6 bash -c 'video_id=$5; \
                                        frame_dir=$0/$1/$2/$3/$4/$5; \
                                        links_dir=$0/action/$1/$2/$3/$4/$5; \
                                        labels_pkl=$test_labels_path; \
                                        modality=$2; \
                                        python3 -m epic_kitchens.preprocessing.split_segments \
                                                   $video_id \
                                                   $frame_dir \
                                                   $links_dir \
                                                   $labels_pkl \
                                                   $modality \
                                                   --frame-format '\''frame_%010d.jpg'\'' \
                                                   --fps 60 \
                                                   --of-stride 2 \
                                                   --of-dilation 3'
done
