from __future__ import print_function, division
import argparse
import json
import pickle
from pathlib import Path
from addict import Dict as adict
from collections import defaultdict
from evaluation import add_min_value
from evaluation import load_tsn_results
from evaluation import load_audio_results
import numpy as np


def to_submission_struct(modalities, results_fpaths, classes_maps, weights):
    test_sets = ['s1', 's2']
    classification_types = ['verb', 'noun']
    num_classes = {'verb': 125, 'noun': 352}
    weights_indices = {'rgb': 0, 'flow': 1, 'audio': 2}

    all_segment_indices = defaultdict(lambda: list())
    loaded_scores = defaultdict(lambda: defaultdict(lambda: list()))
    scores = defaultdict(lambda: dict())

    for c in classification_types:
        for t in test_sets:
            for m in modalities:
                if m == 'audio':
                    segment_indices, s = load_audio_results(results_fpaths[t][c][m])
                else:
                    segment_indices, s = load_tsn_results(results_fpaths[t][c][m], is_challenge=True)
                all_segment_indices[t] = np.union1d(all_segment_indices[t], segment_indices)
                loaded_scores[t][c].append(weights[c][weights_indices[m]]*s)
            scores[t][c] = np.sum(loaded_scores[t][c], axis=0)
            all_segment_indices[t] = all_segment_indices[t].astype(int)

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


def main(modalities_confs, weights, classes_maps):
    modalities = list(modalities_confs.keys())
    test_sets = ['s1', 's2']
    classification_types = ['verb', 'noun']
    results_fpaths = adict()
    for m in modalities:
        for c in classification_types:
            conf = modalities_confs[m][c].conf
            for t in test_sets:
                if m == 'audio':
                    results_fpaths[t][c][m] = '{}/{}_{}_testset_{}_{}_lr_{}_model_{:03d}.csv'.format(conf.checkpoint,
                                                                                                     conf.snapshot_pref,
                                                                                                     conf.type, t,
                                                                                                     conf.arc, conf.lr,
                                                                                                     conf.test_epoch)
                else:
                    results_fpaths[t][c][m] = '{}/tsn_{}_{}_testset_{}_{}_lr_{}_model_{:03d}.npz'.format(
                        conf.checkpoint, conf.snapshot_pref, m, t, conf.arch, conf.lr, conf.test_model)

    submission_results = to_submission_struct(modalities, results_fpaths, classes_maps, weights)
    seen_json_fpath = 'seen_weighted_{}.json'.format('+'.join(modalities))
    with open(seen_json_fpath, 'w') as outfile:
        json.dump(submission_results['s1'], outfile)

    unseen_json_fpath = 'unseen_weighted_{}.json'.format('+'.join(modalities))
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
        with open(weighted_json.modalities[m].verb_conf) as json_file:
            modalities_confs[m].verb.conf = adict(json.load(json_file))
            if m != 'audio':
                if modalities_confs[m].verb.conf.class_type != 'verb':
                    raise Exception('Verb configuration file is not verb type')
            elif modalities_confs[m].verb.conf.type != 'verb':
                raise Exception('Verb configuration file is not verb type')

        with open(weighted_json.modalities[m].noun_conf) as json_file:
            modalities_confs[m].noun.conf = adict(json.load(json_file))
            if m != 'audio':
                if modalities_confs[m].noun.conf.class_type != 'noun':
                    raise Exception('Noun configuration file is not noun type')
            elif modalities_confs[m].noun.conf.type != 'noun':
                raise Exception('Noun configuration file is not noun type')

    classes_maps = adict()
    classes_maps['verb'] = pickle.load(open(weighted_json.classes_maps.verb, "rb"))
    classes_maps['noun'] = pickle.load(open(weighted_json.classes_maps.noun, "rb"))

    main(modalities_confs, weighted_json.weights, classes_maps)
