#!/usr/bin/env python3

import argparse
import pathlib

import pickle
import numpy as np
import IO
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
from addict import Dict as adict

HELP = """\
Filters the action recognition classes (verb, nouns, or verb+nouns) for the EPIC Action Recognition Challenge
"""


def extract_segments(train_labels, classes_map, classification_type='action'):
    segment_ids, labels = [], []
    for index, row in train_labels.iterrows():
        segment_ids.append(index)
        if classification_type == 'action':
            verb_id = row.verb_class
            noun_id = row.noun_class
            labels.append(classes_map[verb_id][noun_id])
        elif classification_type == 'verb':
            verb_id = row.verb_class
            labels.append(classes_map[verb_id])
        else:
            noun_id = row.noun_class
            labels.append(classes_map[noun_id])
    return np.asarray(segment_ids), np.asarray(labels)


def split_action_training_set(train_labels, classes_map):
    segment_ids, labels = extract_segments(train_labels, classes_map)

    labels_count = np.bincount(labels)

    # Classes with only one sample
    one_sample = np.nonzero(labels_count == 1)[0]
    ix = np.isin(labels, one_sample)
    training_indices = segment_ids[ix]

    # Classes with two samples
    two_samples = np.nonzero(labels_count == 2)[0]
    ix = np.isin(labels, two_samples)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    train_index, test_index = next(sss.split(segment_ids[ix], labels[ix]))
    training_indices = np.append(
        training_indices, segment_ids[ix][train_index])
    testing_indices = segment_ids[ix][test_index]

    # Classes with three samples
    three_samples = np.nonzero(labels_count == 3)[0]
    ix = np.isin(labels, three_samples)

    segment_ids_three_samples = segment_ids[ix]
    labels_three_samples = labels[ix]

    np.random.seed(42)
    indices = np.arange(labels_three_samples.size)
    np.random.shuffle(indices)

    splits = defaultdict(list)
    split_indices = defaultdict(int)
    for ind in indices:
        label = labels_three_samples[ind]
        split_id = split_indices[label]
        splits[split_id].append(ind)
        split_indices[label] += 1

    train_index = np.asarray(splits[0])
    validation_index = np.asarray(splits[1])
    test_index = np.asarray(splits[2])

    training_indices = np.append(
        training_indices, segment_ids_three_samples[train_index])
    testing_indices = np.append(
        testing_indices, segment_ids_three_samples[test_index])
    validation_indices = segment_ids_three_samples[validation_index]

    # Classes with more than 3 samples
    more_samples = np.nonzero(labels_count > 3)[0]
    ix = np.isin(labels, more_samples)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=3 / 20, random_state=42)
    segment_ids_more_samples = segment_ids[ix]
    labels_more_samples = labels[ix]
    train_index, test_index = next(
        sss.split(segment_ids_more_samples, labels_more_samples))
    testing_indices = np.append(
        testing_indices, segment_ids_more_samples[test_index])

    segment_ids_trainVal = segment_ids_more_samples[train_index]
    labels_trainVal = labels_more_samples[train_index]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=2 / 17, random_state=42)
    train_index, validation_index = next(
        sss.split(segment_ids_trainVal, labels_trainVal))
    training_indices = np.append(
        training_indices, segment_ids_trainVal[train_index])
    validation_indices = np.append(
        validation_indices, segment_ids_trainVal[validation_index])

    return {'train': training_indices, 'validation': validation_indices, 'test': testing_indices}


def split_training_set(train_labels, classes_map, classification_type):
    segment_ids, labels = extract_segments(
        train_labels, classes_map, classification_type)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=3 / 20, random_state=42)
    train_index, test_index = next(sss.split(segment_ids, labels))

    testing_indices = segment_ids[test_index]

    segment_ids_trainVal = segment_ids[train_index]
    labels_trainVal = labels[train_index]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=2 / 17, random_state=42)
    train_index, validation_index = next(
        sss.split(segment_ids_trainVal, labels_trainVal))
    training_indices = segment_ids_trainVal[train_index]
    validation_indices = segment_ids_trainVal[validation_index]
    return {'train': training_indices, 'validation': validation_indices, 'test': testing_indices}


def baradel_split_sets(train_labels, classes_map):
    # Baradel split the training set is P01...P25
    # Object Level Visual Reasoning in Videos

    segment_ids_trainVal, labels_trainVal = [], []
    testing_indices = []
    for segment_id, row in train_labels.iterrows():
        id_number = int(row.participant_id[1:])
        if id_number < 26:
            segment_ids_trainVal.append(segment_id)

            verb_id = row.verb_class
            labels_trainVal.append(classes_map[verb_id])
        else:
            testing_indices.append(segment_id)

    segment_ids_trainVal = np.asarray(segment_ids_trainVal)
    labels_trainVal = np.asarray(labels_trainVal)
    testing_indices = np.asarray(testing_indices)

    # Removing classes with only one sample
    labels_count = np.bincount(labels_trainVal)

    one_sample = np.nonzero(labels_count == 1)[0]
    ix = np.isin(labels_trainVal, one_sample)
    training_indices = segment_ids_trainVal[ix]

    ix = np.logical_not(ix)
    segment_ids_trainVal = segment_ids_trainVal[ix]
    labels_trainVal = labels_trainVal[ix]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 / 10, random_state=42)
    train_index, validation_index = next(
        sss.split(segment_ids_trainVal, labels_trainVal))

    training_indices = np.append(
        training_indices, segment_ids_trainVal[train_index])
    validation_indices = segment_ids_trainVal[validation_index]
    return {'train': training_indices, 'validation': validation_indices, 'test': testing_indices}


def main(input_fpaths, output_fpaths, classification_type):
    if 'action' in classification_type:

        if classification_type == 'all_actions':
            verbs, nouns, train_labels = IO.action_classes(
                input_fpaths.verbs, input_fpaths.nouns, input_fpaths.train_labels)

            verb_classes_map = IO.build_classes_map(
                train_labels, verbs, 'verb')
            noun_classes_map = IO.build_classes_map(
                train_labels, nouns, 'noun')

            pickle.dump(verb_classes_map, open(
                output_fpaths.verb_classes_map.as_posix(), "wb"))
            pickle.dump(noun_classes_map, open(
                output_fpaths.noun_classes_map.as_posix(), "wb"))
        else:
            verbs, nouns, train_labels = IO.filter_action_classes(
                input_fpaths.verbs, input_fpaths.nouns, input_fpaths.train_labels)

        classes_map = IO.build_action_classes_map(train_labels, verbs, nouns)
        splits = split_action_training_set(train_labels, classes_map)

        pickle.dump(verbs, open(output_fpaths.verbs.as_posix(), "wb"))
        pickle.dump(nouns, open(output_fpaths.nouns.as_posix(), "wb"))
        pickle.dump(train_labels, open(
            output_fpaths.train_labels.as_posix(), "wb"))
        pickle.dump(classes_map, open(
            output_fpaths.classes_map.as_posix(), "wb"))
        pickle.dump(splits, open(output_fpaths.splits.as_posix(), "wb"))

    elif classification_type in ['verb', 'noun']:
        verbs, nouns, train_labels = IO.filter_action_classes(
            input_fpaths.verbs, input_fpaths.nouns, input_fpaths.train_labels)

        categories = verbs if classification_type == 'verb' else nouns
        classes_map = IO.build_classes_map(
            train_labels, categories, classification_type)
        splits = split_training_set(
            train_labels, classes_map, classification_type)

        pickle.dump(classes_map, open(
            output_fpaths.classes_map.as_posix(), "wb"))
        pickle.dump(splits, open(output_fpaths.splits.as_posix(), "wb"))
    else:
        verbs = IO.EpicClass.load_from(input_fpaths.verbs.as_posix())
        train_labels = pickle.load(
            open(input_fpaths.train_labels.as_posix(), 'rb'))

        classes_map = IO.build_classes_map(train_labels, verbs, 'verb')
        splits = baradel_split_sets(train_labels, classes_map)

        pickle.dump(classes_map, open(
            output_fpaths.classes_map.as_posix(), "wb"))
        pickle.dump(splits, open(output_fpaths.splits.as_posix(), "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=HELP, formatter_class=argparse.RawTextHelpFormatter, )
    parser.add_argument('annotations_dir', type=lambda p: pathlib.Path(
        p).absolute(), help='Path to annotations directory')
    parser.add_argument('--type', type=str, dest='classification_type',
                        default='action', help='Classification type')
    args = parser.parse_args()

    annotations_dir = args.annotations_dir
    input_fpaths = adict()
    input_fpaths.verbs = annotations_dir.joinpath('EPIC_verb_classes.csv')
    input_fpaths.nouns = annotations_dir.joinpath('EPIC_noun_classes.csv')
    input_fpaths.train_labels = annotations_dir.joinpath(
        'EPIC_train_action_labels.pkl')

    output_fpaths = adict()
    if args.classification_type == 'all_actions':
        output_fpaths.verbs = annotations_dir.joinpath(
            'EPIC_ARC_all_verbs.pkl')
        output_fpaths.nouns = annotations_dir.joinpath(
            'EPIC_ARC_all_nouns.pkl')
        output_fpaths.train_labels = annotations_dir.joinpath(
            'EPIC_ARC_all_train_labels.pkl')
        output_fpaths.splits = annotations_dir.joinpath(
            'EPIC_ARC_all_splits.pkl')
        output_fpaths.classes_map = annotations_dir.joinpath(
            'EPIC_ARC_all_classes_map.pkl')
        output_fpaths.verb_classes_map = annotations_dir.joinpath(
            'EPIC_ARC_all_verb_classes_map.pkl')
        output_fpaths.noun_classes_map = annotations_dir.joinpath(
            'EPIC_ARC_all_noun_classes_map.pkl')
    elif args.classification_type == 'action':
        output_fpaths.verbs = annotations_dir.joinpath('EPIC_ARC_verbs.pkl')
        output_fpaths.nouns = annotations_dir.joinpath('EPIC_ARC_nouns.pkl')
        output_fpaths.train_labels = annotations_dir.joinpath(
            'EPIC_ARC_train_labels.pkl')
        output_fpaths.splits = annotations_dir.joinpath('EPIC_ARC_splits.pkl')
        output_fpaths.classes_map = annotations_dir.joinpath(
            'EPIC_ARC_classes_map.pkl')
    elif args.classification_type == 'verb':
        output_fpaths.splits = annotations_dir.joinpath(
            'EPIC_ARC_verb_splits.pkl')
        output_fpaths.classes_map = annotations_dir.joinpath(
            'EPIC_ARC_verb_classes_map.pkl')
    elif args.classification_type == 'noun':
        output_fpaths.splits = annotations_dir.joinpath(
            'EPIC_ARC_noun_splits.pkl')
        output_fpaths.classes_map = annotations_dir.joinpath(
            'EPIC_ARC_noun_classes_map.pkl')
    else:
        output_fpaths.splits = annotations_dir.joinpath(
            'EPIC_ARC_Baradel_splits.pkl')
        output_fpaths.classes_map = annotations_dir.joinpath(
            'EPIC_ARC_Baradel_classes_map.pkl')

    main(input_fpaths, output_fpaths, args.classification_type)
