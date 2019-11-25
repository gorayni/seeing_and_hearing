#!/usr/bin/env python3

from __future__ import print_function, division
import argparse
import pathlib
import json
import numpy as np
import tables

HELP = """\
Calculates the statistics (mean, std. deviation, max, and min values) of a spectrograms dataset.
"""


def main(audio_dataset_fpath, stats_fpath):
    audio_dataset = tables.open_file(audio_dataset_fpath, mode='r')

    mean_cum_sum, time_sum = 0., 0.
    dataset_min_value, dataset_max_value = np.inf, -np.inf

    segments = audio_dataset.root.segments
    spectrograms = audio_dataset.root.spectrograms
    num_rows = spectrograms[0].shape[0]

    for i, segment in enumerate(segments):
        mean = segment['spectrogram_mean']
        max_value = segment['spectrogram_max']
        min_value = segment['spectrogram_min']
        size = segment['spectrogram_size']

        if max_value > dataset_max_value:
            dataset_max_value = max_value

        if min_value < dataset_min_value:
            dataset_min_value = min_value

        mean_cum_sum += mean * num_rows
        time_sum += size / num_rows
    mean_value = mean_cum_sum / time_sum

    sample_variance = 0
    num_pixels = num_rows * time_sum - 1
    for segment in segments:
        num_frames = segment['num_frames']
        start_idx = segment['start_idx']
        end_idx = segment['end_idx']
        last_idx = segment['last_frame_idx']
        frames = spectrograms[start_idx:end_idx]

        for i in range(num_frames - 1):
            mean_diff = np.sum(np.power(frames[i] - mean_value, 2))
            sample_variance += mean_diff / num_pixels

        mean_diff = np.sum(np.power(frames[-1][:, :last_idx] - mean_value, 2))
        sample_variance += mean_diff / num_pixels
    std_dev = np.sqrt(sample_variance)

    with open(stats_fpath, 'w') as outfile:
        json.dump({'mean': mean_value, 'std_dev': std_dev, 'max': dataset_max_value, 'min': dataset_min_value}, outfile,
                  sort_keys=True,
                  indent=4, separators=(',', ': '))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=HELP, formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument('dataset_fpath', type=lambda p: str(pathlib.Path(p).absolute()),
                        help='Path to spectrograms dataset')
    parser.add_argument('stats_fpath', type=lambda p: str(pathlib.Path(p).absolute()),
                        help='Output statistics filepath')
    args = parser.parse_args()
    main(args.dataset_fpath, args.stats_fpath)
