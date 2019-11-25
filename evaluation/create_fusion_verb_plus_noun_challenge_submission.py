from __future__ import print_function, division
import argparse
import json
import pickle
from pathlib import Path
from addict import Dict as adict
from collections import defaultdict
from evaluation import add_min_value
from evaluation import load_npz_results
import numpy as np


def to_submission_struct(results_fpaths, classes_maps):

    test_sets = ['s1', 's2']
    classification_types = ['verb', 'noun']
    num_classes = {'verb': 125, 'noun': 352}

    all_segment_indices = defaultdict(lambda: list())
    scores = defaultdict(lambda: dict())

    for c in classification_types:
        for t in test_sets:
            segment_indices, scores_ = load_npz_results(results_fpaths[t][c], is_challenge=True)
            all_segment_indices[t] = np.union1d(all_segment_indices[t], segment_indices).astype(int)
            scores[t][c] = scores_

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


def main(confs, classes_maps):

    test_sets = ['s1', 's2']
    classification_types = ['verb', 'noun']
    results_fpaths = adict()

    for c in classification_types:
        conf = confs[c]
        for t in test_sets:
            results_fpaths[t][c] = './{}/{}_{}_testset_{}_lr_{}_model_{:03d}.npz'.format(conf.checkpoint,
                                                                                            conf.snapshot_pref,
                                                                                            conf.type, t, conf.lr,
                                                                                            conf.test_epoch)

    submission_results = to_submission_struct(results_fpaths, classes_maps)
    seen_json_fpath = 'seen_fusion_verb+noun.json'
    with open(seen_json_fpath, 'w') as outfile:
        json.dump(submission_results['s1'], outfile)

    unseen_json_fpath = 'unseen_fusion_verb+noun.json'
    with open(unseen_json_fpath, 'w') as outfile:
        json.dump(submission_results['s2'], outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=lambda p: str(Path(p).absolute()),
                        help='JSON configuration filepath')

    json_conf = parser.parse_args().conf
    with open(json_conf) as json_file:
        test_json = adict(json.load(json_file))

    confs = adict()
    with open(test_json.confs.verb_conf) as json_file:
        confs.verb = adict(json.load(json_file))
        if confs.verb.type != 'verb':
            raise Exception('Verb configuration file is not verb type')

    with open(test_json.confs.noun_conf) as json_file:
        confs.noun = adict(json.load(json_file))
        if confs.noun.type != 'noun':
            raise Exception('Noun configuration file is not noun type')

    classes_maps = adict()
    classes_maps['verb'] = pickle.load(open(test_json.classes_maps.verb, "rb"))
    classes_maps['noun'] = pickle.load(open(test_json.classes_maps.noun, "rb"))

    main(confs, classes_maps)
