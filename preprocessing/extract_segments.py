import pickle
import pathlib
import csv

import IO
from addict import Dict as adict


NARRATION_COL = 'narration'
START_TS_COL = 'start_timestamp'
STOP_TS_COL = 'stop_timestamp'
VIDEO_ID_COL = 'video_id'
PARTICIPANT_ID_COL = 'participant_id'


def to_csv(annotations, csv_filepath, annotations_type='train'):
    with open(csv_filepath, 'w', newline='') as csvfile:        
        csv_writer = csv.writer(csvfile, delimiter=',')
        
        for i, annotation in enumerate(annotations.itertuples()):
            video_id = getattr(annotation, VIDEO_ID_COL)

            if annotations_type == 'train':
                segment_fname = "{video_id}_{index}_{narration}".format(index=annotation.Index, video_id=video_id, narration=getattr(
                    annotation, NARRATION_COL).strip().lower().replace(' ', '-'))
            else:
                segment_fname = "{video_id}_{index}".format(
                    index=annotation.Index, video_id=video_id)

            participant_id = getattr(annotation, PARTICIPANT_ID_COL)
            start_timestamp = getattr(annotation, START_TS_COL)
            stop_timestamp = getattr(annotation, STOP_TS_COL)

            start_seconds = IO.timestamp_to_seconds(start_timestamp)
            stop_seconds = IO.timestamp_to_seconds(stop_timestamp)
            duration = IO.seconds_to_timestamp(stop_seconds-start_seconds)

            csv_writer.writerow(
                [participant_id, video_id, segment_fname, start_timestamp, stop_timestamp, duration])


annotations_dir = pathlib.Path('annotations').absolute()
input_fpaths = adict()
input_fpaths.verbs = annotations_dir.joinpath('EPIC_verb_classes.csv')
input_fpaths.nouns = annotations_dir.joinpath('EPIC_noun_classes.csv')
input_fpaths.train_labels = annotations_dir.joinpath(
    'EPIC_train_action_labels.pkl')
input_fpaths.s1_test = annotations_dir.joinpath('EPIC_test_s1_timestamps.pkl')
input_fpaths.s2_test = annotations_dir.joinpath('EPIC_test_s2_timestamps.pkl')

all_train_labels = pickle.load(open(input_fpaths.train_labels.as_posix(), 'rb'))
to_csv(all_train_labels, 'annotations/all_action_segments.csv')

filtered_train_labels = IO.filter_action_classes(
    input_fpaths.verbs, input_fpaths.nouns, input_fpaths.train_labels)[2]
to_csv(filtered_train_labels, 'annotations/action_segments.csv')

s1_test = pickle.load(open(input_fpaths.s1_test.as_posix(), "rb"))
to_csv(s1_test, 'annotations/s1_test_segments.csv', 'test')

s2_test = pickle.load(open(input_fpaths.s2_test.as_posix(), "rb"))
to_csv(s2_test, 'annotations/s2_test_segments.csv', 'test')
