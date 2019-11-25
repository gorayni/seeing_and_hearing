import numpy as np
import csv
import os
import sklearn.metrics as metrics
from itertools import product
from collections import namedtuple


EvalResult = namedtuple('EvalResult',
                        ['lr', 'date', 'time', 'epochs', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'filename'])


def load_data_from_logs(classification_type, arc=None, modality=None):
    eval_data = list()
    for log_file in os.listdir('logs'):
        if not log_file.startswith('training_' + classification_type) or not log_file.endswith('.log'):
            continue
        if arc and arc not in log_file:
            continue
        if modality and modality.upper() not in log_file.upper():
            continue

        learning_rate, date, time = log_file[:-4].split('_')[-3:]
        learning_rate = float(learning_rate)
        date = '/'.join([date[:4], date[4:6], date[6:]])
        time = ':'.join([time[:2], time[2:4], time[4:]])

        epochs, train_loss, train_acc, val_loss, val_acc = np.loadtxt('logs/' + log_file, delimiter=' ',
                                                                      usecols=(1, 3, 5, 7, 9), unpack=True)
        evalResult = EvalResult(
            *[learning_rate, date, time, epochs, train_loss, train_acc, val_loss, val_acc, log_file])
        eval_data.append(evalResult)
    eval_data.sort(key=lambda l: l.lr, reverse=True)
    return eval_data


def load_tsn_results(npz_filepath, is_challenge=False):
    results = np.load(npz_filepath)
    segment_indices = results['segment_indices']
    all_scores = np.asarray(results['scores'])
    scores = all_scores.squeeze().mean(axis=1)

    if is_challenge:
        Results = namedtuple('Results', ['segment_indices', 'scores'])
        return Results(segment_indices, scores)

    predictions = scores.argmax(axis=1)
    groundtruth = results['labels']

    Results = namedtuple('Results', ['segment_indices', 'all_scores', 'scores', 'predictions', 'groundtruth'])
    return Results(segment_indices, all_scores, scores, predictions, groundtruth)


def load_npz_results(npz_filepath, is_challenge=False):
    results = np.load(npz_filepath)
    segment_indices = results['segment_indices']
    scores = np.asarray(results['scores'])

    if is_challenge:
        Results = namedtuple('Results', ['segment_indices', 'scores'])
        return Results(segment_indices, scores)

    predictions = scores.argmax(axis=1)
    groundtruth = results['labels']

    Results = namedtuple('Results', ['segment_indices', 'scores', 'predictions', 'groundtruth'])
    return Results(segment_indices, scores, predictions, groundtruth)


def load_audio_results(csv_filepath):
    segment_indices = list()
    scores = list()
    with open(csv_filepath, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        for i, (segment_index, score) in enumerate(reader):
            if i < 1:
                continue
            segment_indices.append(int(segment_index))
            score = np.fromstring(score[1:-1], dtype=float, sep=' ')
            scores.append(score)
    num_segments = len(segment_indices)
    num_categories = scores[0].shape[0]
    scores = np.concatenate(scores).reshape((num_segments, num_categories))
    return segment_indices, scores


def read_results(csv_filepath):
    segment_indices = list()
    scores = list()
    groundtruth = list()

    with open(csv_filepath, 'r') as f:
        reader = csv.reader(f, delimiter=",")
        for i, (segment_index, score, label) in enumerate(reader):
            if i < 1:
                continue
            segment_indices.append(int(segment_index))
            score = np.fromstring(score[1:-1], dtype=float, sep=' ')
            scores.append(score)
            groundtruth.append(int(label))

    segment_indices = np.asarray(segment_indices)
    scores = np.asarray(scores)
    predictions = scores.argmax(axis=1)
    groundtruth = np.asarray(groundtruth)
    Results = namedtuple('Results', ['segment_indices', 'scores', 'predictions', 'groundtruth'])
    return Results(segment_indices, scores, predictions, groundtruth)


def top_n_accuracy(scores, groundtruth, top_n=5):
    top = scores.argsort()[:, -top_n:]
    predictions = list()
    for i, gt in enumerate(groundtruth):
        if gt in top[i, :]:
            predictions.append(gt)
        else:
            predictions.append(top[i, -1])
    return metrics.accuracy_score(groundtruth, predictions)


def measure_performance(scores, groundtruth):
    predictions = scores.argmax(axis=1)
    top1_acc = metrics.accuracy_score(groundtruth, predictions)
    top5_acc = top_n_accuracy(scores, groundtruth, top_n=5)
    avg_precision = metrics.precision_score(groundtruth, predictions, average='macro')
    avg_recall = metrics.recall_score(groundtruth, predictions, average='macro')
    Results = namedtuple('Results', ['top1_acc', 'top5_acc', 'avg_precision', 'avg_recall'])
    return Results(top1_acc, top5_acc, avg_precision, avg_recall)


def random_accuracy_baseline(categories, num_classes, total_num_classes, split='test'):
    rand_acc_baseline = 0.
    if split == 'test':
        for category in categories:
            p_train = num_classes['train'][category] + num_classes['validation'][category]
            p_train /= total_num_classes['train'] + total_num_classes['validation']
            p_test = num_classes['test'][category] / total_num_classes['test']
            rand_acc_baseline += p_train * p_test
    else:
        for category in categories:
            p_train = num_classes['train'][category] / total_num_classes['train']
            p_validation = num_classes['validation'][category] / total_num_classes['validation']
            rand_acc_baseline += p_train * p_validation
    return rand_acc_baseline


def _split_train_test_sets(labels, test_split='test'):
    all_labels = np.hstack([l for l in labels.values()])
    categories = {c: i for i, c in enumerate(np.unique(all_labels))}

    if test_split == 'test':
        train_splits = ['train', 'validation']
    else:
        train_splits = ['train']

    train_set = np.hstack([labels[s] for s in train_splits])
    train_set = np.asarray([categories[t] for t in train_set])

    test_set = labels[test_split]
    test_set = np.asarray([categories[t] for t in test_set])

    return categories, train_set, test_set


def _calculate_probability(num_categories, split_set):
    probability = np.zeros(num_categories)
    labels, counts = np.unique(split_set, return_counts=True)
    num_samples = counts.sum()
    for label, num_occurrences in zip(labels, counts):
        probability[label] = num_occurrences / num_samples
    return probability


def random_accuracy_baseline(labels, test_split='test'):
    categories, train_set, test_set = _split_train_test_sets(labels, test_split)
    num_categories = len(categories)
    train_prob = _calculate_probability(num_categories, train_set)
    test_prob = _calculate_probability(num_categories, test_set)
    return train_prob.dot(test_prob)


def calculate_random_baseline(labels, test_split='test', n=100, seed=42):
    categories, train_set, test_set = _split_train_test_sets(labels, test_split)
    num_categories = len(categories)

    probabilities = _calculate_probability(num_categories, train_set)

    top1_acc = np.zeros(n)
    top5_acc = np.zeros(n)
    avg_precision = np.zeros(n)
    avg_recall = np.zeros(n)

    np.random.seed(seed)
    for i in range(n):
        predictions = np.asarray(
            [np.random.choice(num_categories, size=5, replace=False, p=probabilities) for _ in range(len(test_set))])
        top5_predictions = np.asarray(
            [target if target in predictions[j, :] else -1 for j, target in enumerate(test_set)])

        top1_acc[i] = metrics.accuracy_score(test_set, predictions[:, 0])
        top5_acc[i] = metrics.accuracy_score(test_set, top5_predictions)
        avg_precision[i] = metrics.precision_score(test_set, predictions[:, 0], average='macro')
        avg_recall[i] = metrics.recall_score(test_set, predictions[:, 0], average='macro')

    Results = namedtuple('Results', ['top1_acc', 'top5_acc', 'avg_precision', 'avg_recall'])
    return Results(top1_acc.mean(), top5_acc.mean(), avg_precision.mean(), avg_recall.mean())


def calculate_largest_class_baseline(labels, test_split='test'):
    train_set, test_set = _split_train_test_sets(labels, test_split)[1:]    
    classes, counts = np.unique(train_set, return_counts=True)
    largest_class_indices = np.argsort(counts)
    largest_classes = classes[largest_class_indices]
    
    predictions = np.tile(largest_classes[-5:], (len(test_set), 1))
    top1_acc = metrics.accuracy_score(test_set, predictions[:, -1])

    top5_predictions = np.asarray([target if target in predictions[j, :] else -1 for j, target in enumerate(test_set)])
    top5_acc = metrics.accuracy_score(test_set, top5_predictions)
    avg_precision = metrics.precision_score(test_set, predictions[:, -1], average='macro')
    avg_recall = metrics.recall_score(test_set, predictions[:, -1], average='macro')

    Results = namedtuple('Results', ['top1_acc', 'top5_acc', 'avg_precision', 'avg_recall'])
    return Results(top1_acc, top5_acc, avg_precision, avg_recall)


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def find_weights(weights, groundtruth, *scores):
    best_weights = None
    best_acc = 0.
    num_scores = len(scores)
    scores = np.asarray(scores)

    weights = [weights] * num_scores
    for w in cartesian_product(*weights):
        weighted_scores = scores * w[:, np.newaxis, np.newaxis]
        predictions = weighted_scores.sum(axis=0).argmax(axis=1)

        acc = metrics.accuracy_score(groundtruth, predictions)
        if acc > best_acc:
            best_weights = w
            best_acc = acc
    return best_weights


def to_subcategory_scores(category_dict, scores, num_classes, groundtruth=None):
    num_test_samples, num_original_classes = scores.shape
    new_scores = np.zeros((num_test_samples, num_classes))

    for (i, j) in product(range(num_test_samples), range(num_original_classes)):
        new_scores[i, category_dict[j]] += scores[i, j]

    if groundtruth is not None:
        new_groundtruth = [category_dict[g] for g in groundtruth]
        return new_scores, new_groundtruth
    return new_scores


def add_min_value(scores):
    min_val = np.abs(scores.min(axis=1))
    min_val = min_val[..., np.newaxis]
    return np.add(scores, min_val)


def build_categories_indices(action_classes_map):
    verbs_by_action = dict()
    nouns_by_action = dict()
    for verb_idx, noun_map in action_classes_map.items():
        for noun_idx, action_idx in noun_map.items():
            verbs_by_action[action_idx] = verb_idx
            nouns_by_action[action_idx] = noun_idx
    return verbs_by_action, nouns_by_action


def measure_action_performance(verbs_by_action, nouns_by_action, groundtruth, scores):
    verb_scores, verb_groundtruth = to_subcategory_scores(verbs_by_action, scores, 125, groundtruth)
    noun_scores, noun_groundtruth = to_subcategory_scores(nouns_by_action, scores, 352, groundtruth)

    action_results = measure_performance(scores, groundtruth)
    verb_results = measure_performance(verb_scores, verb_groundtruth)
    noun_results = measure_performance(noun_scores, noun_groundtruth)

    CategoryResults = namedtuple('CategoryResults', ['action', 'verb', 'noun'])
    return CategoryResults(action_results, verb_results, noun_results)


def measure_verb_plus_noun_performance(verb, noun, score_index_by_verb, score_index_by_noun, groundtruth):
    num_test_samples, num_verb_classes = verb.scores.shape
    num_action_classes = np.sum([len(i) for i in score_index_by_noun.values()])

    scores = np.zeros((num_test_samples, num_action_classes))
    for i, j in product(range(num_test_samples), range(num_verb_classes)):
        scores[i, score_index_by_verb[j]] += verb.scores[i, j]

    num_noun_classes = noun.scores.shape[1]
    for i, j in product(range(num_test_samples), range(num_noun_classes)):
        scores[i, score_index_by_noun[j]] += noun.scores[i, j]

    action_results = measure_performance(scores, groundtruth)
    verb_results = measure_performance(verb.scores, verb.groundtruth)
    noun_results = measure_performance(noun.scores, noun.groundtruth)

    CategoryResults = namedtuple('CategoryResults', ['action', 'verb', 'noun'])
    return CategoryResults(action_results, verb_results, noun_results)
