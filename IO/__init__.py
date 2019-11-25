from __future__ import print_function, division
import csv
import pickle
from collections import defaultdict
from itertools import product

import numpy as np


class EpicClass:
    def __init__(self, id_, name, synonyms):
        self.id_ = id_
        self.name = name
        self.synonyms = synonyms

    @staticmethod
    def load_from(class_csv_file):
        classes = list()
        with open(class_csv_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            next(reader, None)
            for row in reader:
                clazz, name, synonyms = row
                clazz = int(clazz)
                synonyms = synonyms[1:-
                                    1].replace(" ", '').replace("'", '').split(',')

                classes.append(EpicClass(clazz, name, synonyms))
        return classes

    def __str__(self):
        return "[id_: {}, name: {}]".format(self.id_, self.name)

    def __repr__(self):
        return "[id_: {}, name: {}]".format(self.id_, self.name)


def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d


def seconds_to_timestamp(total_seconds: float) -> str:
    ss = total_seconds % 60
    mm = np.floor((total_seconds / 60) % 60)
    hh = np.floor((total_seconds / (60 * 60)))
    return "{:02.0f}:{:02.0f}:{:0.3f}".format(hh, mm, ss)


def timestamp_to_seconds(timestamp: str) -> float:
    hours, minutes, seconds = map(float, timestamp.split(":"))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    return total_seconds


def get_duration(row):
    stop_sec = timestamp_to_seconds(row.stop_timestamp)
    start_sec = timestamp_to_seconds(row.start_timestamp)
    return stop_sec - start_sec


def build_action_classes_map(train_labels, verbs, nouns):
    verbs = sorted(verbs, key=lambda x: x.name)
    nouns = sorted(nouns, key=lambda x: x.name)

    verbs_indices = {v.id_: ind for ind, v in enumerate(verbs)}
    nouns_indices = {n.id_: ind for ind, n in enumerate(nouns)}

    num_instances = np.zeros(
        (len(verbs_indices), len(nouns_indices)), dtype=int)

    for index, row in train_labels.iterrows():
        verb_id = row.verb_class
        if verb_id not in verbs_indices:
            continue

        noun_id = row.noun_class
        if noun_id not in nouns_indices:
            continue

        verb_ind = verbs_indices[verb_id]
        noun_ind = nouns_indices[noun_id]
        num_instances[verb_ind, noun_ind] += 1

    verbs_ids = [v.id_ for v in verbs]
    nouns_ids = [n.id_ for n in nouns]

    class_map = defaultdict(lambda: defaultdict(int))
    i = 0
    for verb_id, noun_id in product(verbs_ids, nouns_ids):
        verb_index = verbs_indices[verb_id]
        noun_index = nouns_indices[noun_id]
        if num_instances[verb_index, noun_index] == 0:
            continue
        class_map[verb_id][noun_id] = i
        i += 1
    return default_to_regular(class_map)


def build_classes_map(train_labels, categories, _type):
    categories = sorted(categories, key=lambda x: x.name)
    categories_indices = {c.id_: ind for ind, c in enumerate(categories)}

    num_instances = np.zeros(len(categories_indices), dtype=int)
    for index, row in train_labels.iterrows():
        if _type == 'verb':
            category_id = row.verb_class
        else:
            category_id = row.noun_class
        if category_id not in categories_indices:
            continue
        category_id = categories_indices[category_id]
        num_instances[category_id] += 1

    categories_ids = [c.id_ for c in categories]

    class_map = defaultdict(lambda: defaultdict(int))
    i = 0
    for category_id in categories_ids:
        category_index = categories_indices[category_id]
        if num_instances[category_index] == 0:
            continue
        class_map[category_id] = i
        i += 1
    return default_to_regular(class_map)


def action_classes(verbs_fpath, nouns_fpath, train_labels_fpath):
    verbs = EpicClass.load_from(verbs_fpath.as_posix())
    nouns = EpicClass.load_from(nouns_fpath.as_posix())
    train_labels = pickle.load(open(train_labels_fpath.as_posix(), 'rb'))
    return verbs, nouns, train_labels


def filter_action_classes(verbs_fpath, nouns_fpath, train_labels_fpath):
    verbs = EpicClass.load_from(verbs_fpath.as_posix())
    nouns = EpicClass.load_from(nouns_fpath.as_posix())
    train_labels = pickle.load(open(train_labels_fpath.as_posix(), 'rb'))

    # Filtering verb classes
    verb_classes = train_labels['verb_class'].value_counts()
    verb_classes = verb_classes[verb_classes > 100]

    # Filtering noun classes
    noun_classes = train_labels['noun_class'].value_counts()
    noun_classes = noun_classes[noun_classes > 100]

    # Filtering video segments
    verbs = sorted([verbs[i]
                    for i in verb_classes.keys()], key=lambda x: x.name)
    nouns = sorted([nouns[i]
                    for i in noun_classes.keys()], key=lambda x: x.name)

    train_labels = train_labels[
        (train_labels.verb_class.isin(verb_classes.keys())) & (train_labels.noun_class.isin(noun_classes.keys()))]
    return verbs, nouns, train_labels


def filter_classes(categories_fpath, train_labels_fpath,
                   classification_type='verb'):
    categories = EpicClass.load_from(categories_fpath.as_posix())
    train_labels = pickle.load(open(train_labels_fpath.as_posix(), 'rb'))

    categories_classes = train_labels[classification_type + '_class'].value_counts()
    categories_classes = categories_classes[categories_classes > 100]

    categories = sorted([categories[i] for i in categories_classes.keys()], key=lambda x: x.name)
    return categories


def count_num_classes(classes_map):
    num_classes = 0
    for v in classes_map.values():
        if isinstance(v, dict):
            num_classes += len(v)
        else:
            num_classes += 1
    return num_classes
