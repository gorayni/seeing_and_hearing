from __future__ import print_function, division
import argparse
import json
import pickle
from pathlib import Path
from addict import Dict as adict
from collections import defaultdict
from evaluation import add_min_value
from evaluation import load_npz_results
from evaluation import to_subcategory_scores
from evaluation import build_categories_indices
import numpy as np


def to_submission_struct(results_fpaths, verbs_by_action, nouns_by_action):
    test_sets = ['s1', 's2']
    num_classes = {'verb': 125, 'noun': 352}
    all_segment_indices = defaultdict(lambda: list())
    scores = defaultdict(lambda: dict())

    for t in test_sets:
        segment_indices, fusion_scores = load_npz_results(results_fpaths[t], is_challenge=True)
        all_segment_indices[t] = segment_indices.astype(int)

        scores[t]['action'] = fusion_scores
        scores[t]['verb'] = to_subcategory_scores(verbs_by_action, scores[t]['action'], num_classes['verb'])
        scores[t]['noun'] = to_subcategory_scores(nouns_by_action, scores[t]['action'], num_classes['noun'])

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

        for i, segment_idx in enumerate(all_segment_indices[t]):
            json_scores = {}
            for j in range(100):
                idx = sorted_indices[i, j]
                action_index = '{},{}'.format(verbs_by_action[idx], nouns_by_action[idx])
                json_scores[action_index] = float(scores[t]['action'][i, idx])
            results[str(segment_idx)]['action'] = json_scores

        submission_results[t] = {'version': '0.1', 'challenge': 'action_recognition', 'results': results}
    return submission_results


def main(conf):

    classes_map = pickle.load(open(conf.classes_map, "rb"))
    verbs_by_action, nouns_by_action = build_categories_indices(classes_map)

    results_fpaths = adict()
    for t in ['s1', 's2']:
        results_fpaths[t] = './{}/{}_{}_testset_{}_lr_{}_model_{:03d}.npz'.format(conf.checkpoint,
                                                                                        conf.snapshot_pref,
                                                                                        conf.type, t, conf.lr,
                                                                                        conf.test_epoch)

    submission_results = to_submission_struct(results_fpaths, verbs_by_action, nouns_by_action)
    seen_json_fpath = 'seen_fusion_action.json'
    with open(seen_json_fpath, 'w') as outfile:
        json.dump(submission_results['s1'], outfile, sort_keys=True)

    unseen_json_fpath = 'unseen_fusion_action.json'
    with open(unseen_json_fpath, 'w') as outfile:
        json.dump(submission_results['s2'], outfile, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=lambda p: str(Path(p).absolute()),
                        help='JSON configuration filepath')

    json_conf = parser.parse_args().conf
    with open(json_conf) as json_file:
        conf = adict(json.load(json_file))
        if conf.type != 'action':
            raise Exception('Action configuration file is not action type')

    main(conf)
