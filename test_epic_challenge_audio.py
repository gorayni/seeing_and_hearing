from __future__ import print_function, division
import argparse
import json
from pathlib import Path
from addict import Dict as adict
from torch.autograd import Variable

import torch
import torch.nn.parallel
import torchvision.transforms as transforms
import numpy as np

from IO.dataset import EpicAudioTestSet
from IO.misc import ToFloatTensor


def test(loader, model, cuda, save):
    model.eval()

    segment_ids = list()
    scorez = list()
    for keys, data in loader:
        if cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        scores = model(data)

        scores = scores.data.cpu().numpy()
        for i, key in enumerate(keys):
            segment_ids.append(int(key))
            scorez.append(scores[i,:])
    segment_indices=np.asarray(segment_ids)
    scores=np.asarray(scorez)
    np.savez(save, segment_indices=segment_indices, scores=scores)

def main(conf):
    conf.cuda = conf.cuda and torch.cuda.is_available()
    torch.manual_seed(conf.seed)
    if conf.cuda:
        torch.cuda.manual_seed(conf.seed)
        print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    composed = transforms.Compose([ToFloatTensor()])
    print(conf)

    state = torch.load('./{}/{}_{}_{}_lr_{}_ckpt.t7'.format(conf.checkpoint, conf.snapshot_pref, conf.type, conf.arc, conf.lr))
    epoch = conf.test_epoch if conf.test_epoch else state['epoch']

    print("Testing audio model {} {} {} (epoch {})".format(conf.type, conf.arc, conf.lr, epoch))
    model = torch.load(
        './{}/{}_{}_{}_lr_{}_model_{:03d}.t7'.format(conf.checkpoint,
                                                     conf.snapshot_pref, conf.type, conf.arc, conf.lr, epoch))
    if conf.cuda:
        model = torch.nn.DataParallel(model).cuda()

    for test_set in conf.challenge_test_sets:
        test_dataset = EpicAudioTestSet(test_set.audio_hdf5, transform=composed)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=None,
                                                  num_workers=conf.num_workers, pin_memory=conf.cuda, sampler=None)

        results = './{}/{}_{}_testset_{}_{}_lr_{}_model_{:03d}.npz'.format(conf.checkpoint,
                                                                 conf.snapshot_pref, conf.type, test_set.name,
                                                                              conf.arc, conf.lr, epoch)
        print("Saving results in {}".format(results))
        test(test_loader, model, conf.cuda, save=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=lambda p: str(Path(p).absolute()), help='JSON configuration filepath')

    json_conf = parser.parse_args().conf

    with open(json_conf) as json_file:
        conf = adict(json.load(json_file))

    main(conf)
