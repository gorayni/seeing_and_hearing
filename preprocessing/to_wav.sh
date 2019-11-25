#!/bin/bash

find EPIC_KITCHENS_2018/videos/* -mindepth 1 -type d | xargs -I {} bash -c 'audio_dir=`echo {} | tr "/" " " | awk '\''{print $1"/audio/"$3"/"$4}'\''`; mkdir -p $audio_dir'

find EPIC_KITCHENS_2018/videos -mindepth 3 -type f -name '*.MP4' | parallel -P25 -I {} ./files_to_wav.sh {}
