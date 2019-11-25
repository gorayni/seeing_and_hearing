from __future__ import print_function, division
import argparse
import json
from pathlib import Path
from addict import Dict as adict
from torch.autograd import Variable
from IO.dataset import EpicSegment
import pickle
import os
import torch
import torch.nn.parallel
import torchvision.transforms as transforms
import numpy as np

from IO.dataset import EpicAudioTestSet
from IO.misc import ToFloatTensor


fc_output = None


def fc_output_hook(module, input_, output):
    global fc_output
    fc_output = output


def extract(model, segment_index, spectrogram, cuda):
    global fc_output

    with torch.no_grad():
        spectrogram = torch.autograd.Variable(
            spectrogram.view(-1,  spectrogram.size(1), spectrogram.size(2)))
        model(spectrogram)
        segment_index = segment_index.cpu().numpy().copy()[0]
        fc_output = np.squeeze(fc_output.cpu().numpy().copy())
    return segment_index, fc_output


def main(conf, encodings, test_set):

    conf.cuda = conf.cuda and torch.cuda.is_available()
    torch.manual_seed(conf.seed)
    if conf.cuda:
        torch.cuda.manual_seed(conf.seed)
        print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    composed = transforms.Compose([ToFloatTensor()])
    print(conf)

    state = torch.load(
        './{}/{}_{}_{}_lr_{}_ckpt.t7'.format(conf.checkpoint, conf.snapshot_pref, conf.type, conf.arc, conf.lr))
    epoch = conf.test_epoch if conf.test_epoch else state['epoch']

    print("Extracting audio model {} {} {} (epoch {})".format(
        conf.type, conf.arc, conf.lr, epoch))
    if not conf.weights_file:
        model = torch.load('./{}/{}_{}_{}_lr_{}_model_{:03d}.t7'.format(
            conf.checkpoint, conf.snapshot_pref, conf.type, conf.arc, conf.lr, epoch))
    else:
        model = torch.load(conf.weights_file)

    if conf.cuda:
        model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(encodings):
        encoded_dataset = pickle.load(open(encodings, "rb"))
    else:
        encoded_dataset = dict()

    if test_set:
        dataset = EpicAudioTestSet(test_set.audio_hdf5, transform=composed)
    else:
        dataset = EpicAudioTestSet(conf.audio_hdf5, transform=composed)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=None, num_workers=1, pin_memory=conf.cuda, sampler=None)

    model.eval()

    is_vgg11 = conf.arc == 'VGG11'
    is_train = not test_set

    verb_id = None
    noun_id = None
    model.module.classifier[3].register_forward_hook(fc_output_hook)
    for segment_id, spectrogram in loader:
        segment_index, fc_output = extract(
            model, segment_id, spectrogram, conf.cuda)
        if segment_index not in encoded_dataset:
            if is_train:
                verb_id, noun_id = loader.dataset.get_original_labels(
                    segment_index)
            epic_segment = EpicSegment(
                segment_id, verb_id=verb_id, noun_id=noun_id)

            if is_vgg11:
                epic_segment.audio_vgg = fc_output
            else:
                epic_segment.audio_traddil = fc_output

            encoded_dataset[segment_index] = epic_segment
        elif is_vgg11:
            encoded_dataset[segment_index].audio_vgg = fc_output
        else:
            encoded_dataset[segment_index].audio_traddil = fc_output

    pickle.dump(encoded_dataset, open(encodings, "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=lambda p: str(
        Path(p).absolute()), help='JSON configuration filepath')
    parser.add_argument('encodings', type=lambda p: Path(p).absolute(),
                        help='Encodings filepath')
    parser.add_argument('-test_set', type=str, default=None,
                        help='Test set: either s1 or s2')

    args = parser.parse_args()
    json_conf = args.conf

    with open(json_conf) as json_file:
        conf = adict(json.load(json_file))

    test_set = None
    if args.test_set:
        for ts in conf.challenge_test_sets:
            if ts.name.lower() == args.test_set.lower():
                test_set = ts
                break
    main(conf, args.encodings, test_set)
