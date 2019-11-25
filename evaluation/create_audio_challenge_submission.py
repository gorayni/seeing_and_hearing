from __future__ import print_function, division
import argparse
import json
import pickle
from pathlib import Path
from addict import Dict as adict
from evaluation import add_min_value
from evaluation import load_audio_results
from collections import defaultdict
import numpy as np


def to_submission_struct(results_fpaths, classes_maps):

    test_sets = ['s1', 's2']
    classification_types = ['verb', 'noun']

    all_segment_indices = defaultdict(lambda: list())
    scores = defaultdict(lambda: dict())
    for t in test_sets:
        for c in classification_types:

            segment_indices, loaded_scores = load_audio_results(results_fpaths[t][c])
            all_segment_indices[t] = np.union1d(all_segment_indices[t], segment_indices)
            scores[t][c] = loaded_scores

        all_segment_indices[t] = all_segment_indices[t].astype(int)

    num_classes = {'verb': 125, 'noun': 352}
    submission_results = dict()
    for t in test_sets:
        results = dict()
        for segment_idx in all_segment_indices[t]:
            results[str(segment_idx)] = dict()

        for c in classification_types:
            scores_ = add_min_value(scores[t][c])
            for i, segment_idx in enumerate(all_segment_indices[t]):
                json_scores = {str(j): 0. for j in range(num_classes[c])}
                for epic_idx, model_idx in classes_maps[c].items():
                    json_scores[str(epic_idx)] = float(scores_[i, model_idx])
                results[str(segment_idx)][c] = json_scores
        submission_results[t] = {'version': '0.1', 'challenge': 'action_recognition', 'results': results}
    return submission_results


def main(confs):

    classification_types = ['verb', 'noun']
    results_fpaths = adict()
    classes_maps = adict()
    for c in classification_types:
        for test_set in confs[c].challenge_test_sets:
            results_fpaths[test_set.name][c] = './{}/audio_{}_testset_{}_{}_lr_{}_model_{:03d}.csv'.format(
                confs[c].checkpoint, confs[c].type, test_set.name, confs[c].arc, confs[c].lr, confs[c].test_epoch)
        classes_maps[c] = pickle.load(open(confs[c].classes_map, "rb"))

    submission_results = to_submission_struct(results_fpaths, classes_maps)

    seen_json_fpath = 'seen_verb_{}_{}_{}_noun_{}_{}_{}.json'.format(confs['verb'].arc, confs['verb'].lr,
                                                                     confs['verb'].test_epoch, confs['noun'].arc,
                                                                     confs['noun'].lr, confs['noun'].test_epoch)
    with open(seen_json_fpath, 'w') as outfile:
        json.dump(submission_results['s1'], outfile)

    unseen_json_fpath = 'seen_verb_{}_{}_{}_noun_{}_{}_{}.json'.format(confs['verb'].arc, confs['verb'].lr,
                                                                     confs['verb'].test_epoch, confs['noun'].arc,
                                                                     confs['noun'].lr, confs['noun'].test_epoch)
    with open(unseen_json_fpath, 'w') as outfile:
        json.dump(submission_results['s2'], outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('noun_conf', type=lambda p: str(Path(p).absolute()),
                        help='Noun JSON configuration filepath')
    parser.add_argument('verb_conf', type=lambda p: str(Path(p).absolute()),
                        help='Verb JSON configuration filepath')
    noun_json_conf = parser.parse_args().noun_conf
    verb_json_conf = parser.parse_args().verb_conf

    with open(noun_json_conf) as json_file:
        noun_conf = adict(json.load(json_file))

    with open(verb_json_conf) as json_file:
        verb_conf = adict(json.load(json_file))

    if verb_conf.type != 'verb':
        raise Exception('Verb configuration file is not verb type')

    if noun_conf.type != 'noun':
        raise Exception('Noun configuration file is not noun type')

    confs = adict()
    confs.verb = verb_conf
    confs.noun = noun_conf
    main(confs)
