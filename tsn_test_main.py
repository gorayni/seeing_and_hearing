from __future__ import print_function, division

import os
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


add_path(os.path.realpath('tsn-pytorch'))

from transforms import ToTorchFormatTensor
from transforms import Stack
from transforms import GroupScale
from transforms import GroupNormalize
from transforms import GroupCenterCrop
from transforms import GroupOverSample
from models import TSN
from IO.dataset import EpicTSNDataset
from IO import count_num_classes
import argparse
import json
import numpy as np
import pathlib
import pickle
import time
import torch
import torch.nn.parallel
import torch.optim
import torchvision
from addict import Dict as adict
from epic_kitchens.dataset.epic_dataset import EpicVideoDataset
from epic_kitchens.dataset.epic_dataset import EpicVideoFlowDataset
from pathlib import Path
from sklearn.metrics import accuracy_score


def eval_video(conf, video_data, net):
    i, segment_index, data, label = video_data
    num_crop = conf.test_crops

    if conf.modality == 'RGB':
        length = 3
    elif conf.modality == 'Flow':
        length = 10
    elif conf.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality " + conf.modality)

    with torch.no_grad():
        input_var = torch.autograd.Variable(
            data.view(-1, length, data.size(2), data.size(3)))
        rst = net(input_var).data.cpu().numpy().copy()
    return i, segment_index[0], rst.reshape((num_crop, conf.test_segments, conf.num_classes)).mean(axis=0).reshape(
        (conf.test_segments, 1, conf.num_classes)), label[0]


def main(conf, test_part=-1, split_name='test'):
    gulp_path = os.path.realpath(conf.gulp_dir)
    gulp_path = Path(gulp_path)

    splits = pickle.load(open(conf.splits, "rb"))
    classes_map = pickle.load(open(conf.classes_map, "rb"))
    conf.num_classes = count_num_classes(classes_map)

    net = TSN(conf.num_classes, 1, conf.modality, base_model=conf.arch, consensus_type=conf.crop_fusion_type,
              dropout=conf.dropout)

    checkpoint = torch.load(conf.weights)
    print("Model epoch {} best prec@1: {}".format(
        checkpoint['epoch'], checkpoint['best_prec1']))

    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(
        checkpoint['state_dict'].items())}
    net.load_state_dict(base_dict)

    if conf.test_crops == 1:
        cropping = torchvision.transforms.Compose([
            GroupScale(net.scale_size),
            GroupCenterCrop(net.input_size),
        ])
    elif conf.test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(net.input_size, net.scale_size)
        ])
    else:
        raise ValueError(
            "Only 1 and 10 crops are supported while we got {}".format(conf.test_crops))

    class_type = 'verb+noun' if conf.class_type == 'action' else conf.class_type
    if conf.modality == 'Flow':
        dataset = EpicVideoFlowDataset(
            gulp_path=gulp_path, class_type=class_type)
    else:
        dataset = EpicVideoDataset(gulp_path=gulp_path, class_type=class_type)

    data_loader = torch.utils.data.DataLoader(
        EpicTSNDataset(dataset, classes_map, splits, num_segments=conf.test_segments,
                       new_length=1 if conf.modality == "RGB" else 5, modality=conf.modality,
                       transform=torchvision.transforms.Compose([cropping, Stack(roll=conf.arch == 'BNInception'),
                                                                 ToTorchFormatTensor(
                                                                     div=conf.arch != 'BNInception'),
                                                                 GroupNormalize(net.input_mean, net.input_std), ]),
                       split_name=split_name, classification_type=conf.class_type, part=test_part), batch_size=1,
        shuffle=False, num_workers=conf.workers * 2, pin_memory=True)

    net = torch.nn.DataParallel(net, device_ids=conf.gpus).cuda()
    net.eval()

    total_num = len(data_loader.dataset)
    output = []

    proc_start_time = time.time()
    for i, (keys, input_, target) in enumerate(data_loader):
        rst = eval_video(conf, (i, keys, input_, target), net)
        output.append(rst[1:])
        cnt_time = time.time() - proc_start_time
        print('video {} done, total {}/{}, average {} sec/video'.format(i,
                                                                        i + 1, total_num, float(cnt_time) / (i + 1)))

    video_index = [x[0] for x in output]
    scores = [x[1] for x in output]
    video_pred = [np.argmax(np.mean(x[1], axis=0)) for x in output]
    video_labels = [x[2] for x in output]

    print('Accuracy {:.02f}%'.format(
        accuracy_score(video_labels, video_pred) * 100))

    if conf.save_scores is not None:
        if split_name != 'test':
            conf.save_scores = conf.save_scores.replace(
                '.npz', '_{}.npz'.format(split_name))

        if test_part > 0:
            conf.save_scores = conf.save_scores.replace(
                '.npz', '_part-{}.npz'.format(test_part))
        np.savez(conf.save_scores, segment_indices=video_index,
                 scores=scores, labels=video_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=lambda p: pathlib.Path(p).absolute(),
                        help='JSON configuration filepath')
    parser.add_argument('-part', type=int, default=-1,
                        help='Test part to evaluate')
    parser.add_argument('-split_name', type=str,
                        default='test', help='Split to evaluate')
    args = parser.parse_args()
    json_conf = args.conf
    test_part = args.part
    with open(json_conf) as json_file:
        conf = adict(json.load(json_file))

    main(conf, test_part, args.split_name)
