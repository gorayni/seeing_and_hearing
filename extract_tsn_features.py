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
from IO.dataset import EpicTSNTestDataset
from IO.dataset import EpicSegment
from IO import count_num_classes
import argparse
import json
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torchvision
from addict import Dict as adict
from epic_kitchens.dataset.epic_dataset import EpicVideoDataset
from epic_kitchens.dataset.epic_dataset import EpicVideoFlowDataset
from pathlib import Path
import pickle


avgpool_output = None
def avgpool_output_hook(module, input_, output):
    global avgpool_output
    avgpool_output = output


def extract(conf, video_data, net):
    global avgpool_output

    i, segment_index, data = video_data

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
        net(input_var)

        avgpool_output = np.squeeze(avgpool_output.cpu().numpy().copy())

    return i, segment_index.cpu().numpy().copy()[0], avgpool_output


def main(conf, encodings, test_set=None, part=-1):

    is_train = not test_set
    gulp_dir = '/'.join(conf.gulp_dir.split('/')[:3])
    if is_train:
        gulp_path = os.path.join(gulp_dir, conf.modality.lower(), 'train')
    else:
        gulp_path = os.path.join(gulp_dir, conf.modality.lower(), 'test', test_set)
    gulp_path = os.path.realpath(gulp_path)

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

    if os.path.isfile(encodings):
        encoded_dataset = pickle.load(open(encodings, "rb"))
    else:
        encoded_dataset = dict()

    data_loader = torch.utils.data.DataLoader(
        EpicTSNTestDataset(dataset, classes_map, num_segments=conf.test_segments,
                           new_length=1 if conf.modality == "RGB" else 5, modality=conf.modality,
                           transform=torchvision.transforms.Compose([cropping, Stack(roll=conf.arch == 'BNInception'),
                                                                     ToTorchFormatTensor(
                                                                         div=conf.arch != 'BNInception'),
                                                                     GroupNormalize(net.input_mean, net.input_std), ]),
                           part=part), batch_size=1, shuffle=False,
        num_workers=conf.workers * 2, pin_memory=True)

    net = torch.nn.DataParallel(net, device_ids=conf.gpus).cuda()
    net.eval()

    net.module.base_model.avgpool.register_forward_hook(avgpool_output_hook)
    total_num = len(data_loader.dataset)
    verb_id, noun_id = None, None
    is_rgb = conf.modality.lower() == 'rgb'

    for i, (keys, input_) in enumerate(data_loader):
        i, segment_id, layer_output = extract(conf, (i, keys, input_), net)

        if segment_id not in encoded_dataset:
            if is_train:
                verb_id, noun_id = data_loader.dataset.get_original_labels(segment_id)
            epic_segment = EpicSegment(segment_id, verb_id=verb_id, noun_id=noun_id)

            if is_rgb:
                epic_segment.rgb = layer_output
            else:
                epic_segment.flow = layer_output
            encoded_dataset[segment_id] = epic_segment
        elif is_rgb:
            encoded_dataset[segment_id].rgb = layer_output
        else:
            encoded_dataset[segment_id].flow = layer_output
        print('video {} done, total {}/{}'.format(i, i + 1, total_num))

    pickle.dump(encoded_dataset, open(encodings, "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=lambda p: Path(p).absolute(),
                        help='JSON configuration filepath')
    parser.add_argument('encodings', type=lambda p: Path(p).absolute(),
                        help='Encodings filepath')
    parser.add_argument('-part', type=int, default=-1,
                        help='Part to evaluate')
    parser.add_argument('-test_set', type=str, default=None,
                        help='Test set: either s1 or s2')

    args = parser.parse_args()
    json_conf = args.conf
    test_part = args.part
    with open(json_conf) as json_file:
        conf = adict(json.load(json_file))

    main(conf, args.encodings, args.test_set, args.part)
