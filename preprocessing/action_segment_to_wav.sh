#!/bin/bash

participant_id=$1;
video_id=$2;
start_time=$4
end_time=$5
duration=$6
data_split=$7

audio_dir="EPIC_KITCHENS_2018/audio/$data_split"
input=$audio_dir/$participant_id/$video_id.wav
action_audio_dir=EPIC_KITCHENS_2018/action/audio/$data_split/$participant_id
output=$action_audio_dir/$3.wav

mkdir -p "$action_audio_dir"

test ! -f $output && ffmpeg -ss $start_time -i $input -t $duration -acodec pcm_s16le -ar 16000 $output
