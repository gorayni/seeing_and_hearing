#!/usr/bin/env python3

import argparse
import os
import pathlib
import pickle
import json
import tables
import audio
from addict import Dict as adict
from IO import get_duration

HELP = """\
Creates and stores the spectrograms from the action segments
"""


class ActionSegment(tables.IsDescription):
    participant_id = tables.StringCol(3)
    video_id = tables.StringCol(6)
    narration = tables.StringCol(77)
    verb_class = tables.Int32Col()
    noun_class = tables.Int32Col()
    start_idx = tables.Int32Col()
    end_idx = tables.Int32Col()
    last_frame_idx = tables.Int32Col()
    spectrogram_mean = tables.Float32Col()
    spectrogram_max = tables.Float32Col()
    spectrogram_min = tables.Float32Col()
    spectrogram_size = tables.Int32Col()
    duration = tables.Float32Col()
    index = tables.Int32Col()
    num_frames = tables.Int32Col()


class TestActionSegment(tables.IsDescription):
    participant_id = tables.StringCol(3)
    video_id = tables.StringCol(6)
    start_idx = tables.Int32Col()
    end_idx = tables.Int32Col()
    last_frame_idx = tables.Int32Col()
    spectrogram_mean = tables.Float32Col()
    spectrogram_max = tables.Float32Col()
    spectrogram_min = tables.Float32Col()
    spectrogram_size = tables.Int32Col()
    duration = tables.Float32Col()
    index = tables.Int32Col()
    num_frames = tables.Int32Col()


def get_audio_fpath(audio_dir, index, row, is_test=False):
    if is_test:
        segment_fname = "{}_{}.wav".format(row.video_id, index)
    else:
        narration = row.narration.strip().lower().replace(' ', '-')
        segment_fname = "{}_{}_{}.wav".format(row.video_id, index, narration)
    return os.path.join(audio_dir, row.participant_id, segment_fname)


def main(args):
    segments = pickle.load(open(args.segments, "rb"))

    hdf5_fpath = os.path.join(args.audio_dir, args.hdf5_fname)
    hdf5_file = tables.open_file(hdf5_fpath, mode='w')
    spectrograms = hdf5_file.create_earray(hdf5_file.root, 'spectrograms', tables.Float64Atom(
    ), shape=(0, 331, 62 * args.num_secs), filters=tables.Filters(complevel=0))
    if args.is_test:
        table_segments = hdf5_file.create_table(
            '/', 'segments', TestActionSegment, "Test action segments")
    else:
        table_segments = hdf5_file.create_table(
            '/', 'segments', ActionSegment, "Action segments labels")

    if args.stats_json:
        with open(args.stats_json) as json_file:
            stats = adict(json.load(json_file))
        std_min = (stats.min - stats.mean) / stats.std_dev
        std_max = (stats.max - stats.mean) / stats.std_dev
    elif args.std_stats_json:
        with open(args.std_stats_json) as json_file:
            stats = adict(json.load(json_file))
    elif args.norm_stats_json:
        with open(args.norm_stats_json) as json_file:
            stats = adict(json.load(json_file))

    start_idx = 0
    segment = table_segments.row
    for index, row in segments.iterrows():

        action_fpath = get_audio_fpath(args.audio_dir, index, row, args.is_test)
        s = audio.extract_spectrogram(action_fpath, 0.03, 0.015, 661)

        if args.stats_json:
            # Standardization
            s = (s - stats.mean) / stats.std_dev
            # Normalization
            s = (s - std_min) / (std_max - std_min)
        elif args.std_stats_json:
            # Standardization
            s = (s - stats.mean) / stats.std_dev
        elif args.norm_stats_json:
            # Normalization
            s = (s - stats.min) / (stats.max - stats.min)

        frames, last_idx = audio.to_frames(s, 62 * args.num_secs)
        for frame in frames:
            spectrograms.append(frame[None])

        if not args.is_test:
            segment['narration'] = row.narration
            segment['verb_class'] = row.verb_class
            segment['noun_class'] = row.noun_class

        segment['participant_id'] = row.participant_id
        segment['video_id'] = row.video_id
        segment['num_frames'] = len(frames)
        segment['start_idx'] = start_idx
        end_idx = start_idx + len(frames)
        segment['end_idx'] = end_idx
        segment['last_frame_idx'] = last_idx
        segment['spectrogram_mean'] = s.mean()
        segment['spectrogram_max'] = s.max()
        segment['spectrogram_min'] = s.min()
        segment['spectrogram_size'] = s.size
        segment['duration'] = get_duration(row)
        segment['index'] = index
        segment.append()

        start_idx = end_idx

    table_segments.flush()
    hdf5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=HELP, formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument('segments', type=lambda p: str(pathlib.Path(p).absolute()),
                        help='Path to action segments pickle file')
    parser.add_argument('audio_dir', type=lambda p: str(pathlib.Path(p).absolute()),
                        help='Dirpath of the action segment WAV files')
    parser.add_argument('--hdf5_fname', type=str, default='audio_dataset.hdf5',
                        help='PyTables HDF5 filename')
    parser.add_argument('--num_secs', type=int, default=4,
                        help='Number of seconds to be divided in the Spectrogram')
    parser.add_argument('--stats_json', type=lambda p: str(pathlib.Path(p).absolute()),
                        help='Statistics JSON filepath')
    parser.add_argument('--norm_stats_json', type=lambda p: str(pathlib.Path(p).absolute()),
                        help='Normalization statistics JSON filepath')
    parser.add_argument('--std_stats_json', type=lambda p: str(pathlib.Path(p).absolute()),
                        help='Standardization statistics JSON filepath')
    parser.add_argument('--test', dest='is_test', action='store_true')
    main(parser.parse_args())
