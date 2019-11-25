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

from IO.dataset import EpicEmbeddingsTestDataset
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
            scorez.append(scores[i, :])

    if save is not None:
        np.savez(save, segment_indices=np.asarray(segment_ids), scores=np.asarray(scorez))


def main(conf):

    conf.cuda = conf.cuda and torch.cuda.is_available()
    torch.manual_seed(conf.seed)
    if conf.cuda:
        torch.cuda.manual_seed(conf.seed)
        print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    composed = transforms.Compose([ToFloatTensor()])
    print(conf)

    state = torch.load('./{}/{}_{}_lr_{}_ckpt.t7'.format(conf.checkpoint, conf.snapshot_pref, conf.type, conf.lr))
    epoch = conf.test_epoch if conf.test_epoch else state['epoch']

    print("Testing fusion model {} {} {} (epoch {})".format(conf.type, conf.arc, conf.lr, epoch))
    model = torch.load(
        './{}/{}_{}_lr_{}_model_{:03d}.t7'.format(conf.checkpoint, conf.snapshot_pref, conf.type, conf.lr, epoch))

    if conf.cuda:
        model = torch.nn.DataParallel(model).cuda()

    for test_set in conf.challenge_test_sets:
        test_dataset = EpicEmbeddingsTestDataset(test_set.embeddings_pkl, transform=composed, classification_type=conf.type,
                                                 num_segments=conf.k, num_embeddings=conf.test_segments,
                                                 modalities=conf.modalities)

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=None,
                                                  num_workers=1, pin_memory=conf.cuda, sampler=None)

        results = './{}/{}_{}_testset_{}_lr_{}_model_{:03d}.npz'.format(conf.checkpoint, conf.snapshot_pref, conf.type,
                                                                        test_set.name, conf.lr, epoch)
        print("Saving results in {}".format(results))
        test(test_loader, model, conf.cuda, save=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=lambda p: str(Path(p).absolute()), help='JSON configuration filepath')

    json_conf = parser.parse_args().conf

    with open(json_conf) as json_file:
        conf = adict(json.load(json_file))

    main(conf)
