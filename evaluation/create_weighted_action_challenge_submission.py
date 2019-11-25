from __future__ import print_function, division
import argparse
import json
import pickle
from pathlib import Path
from addict import Dict as adict
from collections import defaultdict
from evaluation import add_min_value
from evaluation import load_tsn_results
from evaluation import load_npz_results
from evaluation import to_subcategory_scores
from evaluation import build_categories_indices
import numpy as np


def to_submission_struct(modalities, results_fpaths, verbs_by_action, nouns_by_action, weights):

    test_sets = ['s1', 's2']
    weights_indices = {'rgb': 0, 'flow': 1, 'audio': 2}
    num_classes = {'verb': 125, 'noun': 352}

    load_results = {'rgb': load_tsn_results,
                    'flow': load_tsn_results,
                    'audio': load_npz_results}

    all_segment_indices = defaultdict(lambda: list())
    weighted_scores = defaultdict(lambda: list())
    scores = defaultdict(lambda: dict())

    for t in test_sets:
        for m in modalities:
            segment_indices, s = load_results[m](results_fpaths[t][m], is_challenge=True)
            all_segment_indices[t] = np.union1d(all_segment_indices[t], segment_indices)
            weighted_scores[t].append(weights[weights_indices[m]]*s)
        action_scores = np.sum(weighted_scores[t], axis=0)

        scores[t]['action'] = action_scores
        scores[t]['verb'] = to_subcategory_scores(verbs_by_action, action_scores, num_classes['verb'])
        scores[t]['noun'] = to_subcategory_scores(nouns_by_action, action_scores, num_classes['noun'])
        all_segment_indices[t] = all_segment_indices[t].astype(int)

    classification_types = ['verb', 'noun']
    submission_results = dict()
    for t in test_sets:
        results = dict()
        for segment_idx in all_segment_indices[t]:
            results[str(segment_idx)] = dict()

        sorted_indices = np.argsort(scores[t]['action'], axis=1)
        sorted_indices = sorted_indices[:, -100:]

        for i, segment_idx in enumerate(all_segment_indices[t]):
            for c in classification_types:
                json_scores = {str(j): float(scores[t][c][i, j]) for j in range(num_classes[c])}
                results[str(segment_idx)][c] = json_scores

            json_scores = {}
            for j in range(100):
                idx = sorted_indices[i, j]
                action_index = '{},{}'.format(verbs_by_action[idx], nouns_by_action[idx])
                json_scores[action_index] = float(scores[t]['action'][i, idx])
            results[str(segment_idx)]['action'] = json_scores

        submission_results[t] = {'version': '0.1', 'challenge': 'action_recognition', 'results': results}
    return submission_results


def main(modalities_confs, weights, classes_map):
    verbs_by_action, nouns_by_action = build_categories_indices(classes_map)

    modalities = list(modalities_confs.keys())
    test_sets = ['s1', 's2']
    results_fpaths = adict()
    for m in modalities:
        conf = modalities_confs[m].conf
        for t in test_sets:
            if m == 'audio':
                results_fpaths[t][m] = '{}/{}_{}_testset_{}_{}_lr_{}_model_{:03d}.npz'.format(conf.checkpoint,
                                                                                                 conf.snapshot_pref,
                                                                                                 conf.type, t,
                                                                                                 conf.arc, conf.lr,
                                                                                                 conf.test_epoch)
            else:
                results_fpaths[t][m] = '{}/tsn_{}_{}_testset_{}_{}_lr_{}_model_{:03d}.npz'.format(conf.checkpoint,
                                                                                                  conf.snapshot_pref,
                                                                                                  m, t,
                                                                                                  conf.arch, conf.lr,
                                                                                                  conf.test_model)

    submission_results = to_submission_struct(modalities, results_fpaths, verbs_by_action, nouns_by_action, weights)
    seen_json_fpath = 'seen_action_weighted_{}.json'.format('+'.join(modalities))
    with open(seen_json_fpath, 'w') as outfile:
        json.dump(submission_results['s1'], outfile)

    unseen_json_fpath = 'unseen_action_weighted_{}.json'.format('+'.join(modalities))
    with open(unseen_json_fpath, 'w') as outfile:
        json.dump(submission_results['s2'], outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weighted_conf', type=lambda p: str(Path(p).absolute()),
                        help='Weighted JSON configuration filepath')

    weighted_json_conf = parser.parse_args().weighted_conf
    with open(weighted_json_conf) as json_file:
        weighted_json = adict(json.load(json_file))

    modalities_confs = adict()
    modalities = list(weighted_json.modalities.keys())
    for m in modalities:
        with open(weighted_json.modalities[m].conf) as json_file:
            modalities_confs[m].conf = adict(json.load(json_file))
            if m != 'audio':
                if modalities_confs[m].conf.class_type != 'action':
                    raise Exception('Action configuration file is not action type')
            elif modalities_confs[m].conf.type != 'action':
                raise Exception('Action configuration file is not action type')

    classes_map = pickle.load(open(weighted_json.classes_map, "rb"))
    main(modalities_confs, weighted_json.weights, classes_map)
