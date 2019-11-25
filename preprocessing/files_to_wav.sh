#!/bin/bash

filepath=$1
fname=$(basename $1);
fbname=${fname%.*};
audio_fname=`echo $filepath | tr "/" " " | awk '{print $1"/audio/"$3"/"$4}'`/$fbname.wav;
ffmpeg -i $filepath -acodec pcm_s16le $audio_fname
