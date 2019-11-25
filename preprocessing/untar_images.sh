#!/bin/bash

find EPIC_KITCHENS_2018/frames_rgb_flow -mindepth 3 -maxdepth 4 -name '*.tar' | xargs -P25 -I{} bash -c 'userdir=$(dirname {});
													 fname=$(basename {});
													 videoname=${fname%.*};
													 new_dir=$userdir/$videoname;
													 mkdir -p $new_dir;
													 tar xvf {} -C $new_dir;
													 rm {}'
