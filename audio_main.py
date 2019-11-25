from __future__ import print_function, division
import argparse
import json
import os
import pickle
from pathlib import Path
from addict import Dict as adict
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

from IO import count_num_classes
from IO.dataset import EpicAudioDataset
from IO.misc import ToFloatTensor
from models.audio import VGG
from models.audio import TraditionalDilated


def train(loader, model, optimizer, epoch, cuda, log_interval, weight=None, verbose=True):
    model.train()
    global_epoch_loss, n_correct, n_total = 0., 0, 0
    for batch_idx, (_, batch, target) in enumerate(loader):
        criterion = nn.CrossEntropyLoss(weight=weight)
        if cuda:
            batch, target = batch.cuda(), target.cuda()
            criterion = criterion.cuda()

        batch, target = Variable(batch), Variable(target)
        optimizer.zero_grad()
        output = model(batch)

        predictions = torch.max(output, 1)[1].view(target.size())

        n_correct += (predictions.data == target.data).cpu().sum()
        n_total += batch.size()[0]
        train_acc = 100. * float(n_correct)/n_total

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        global_epoch_loss += float(loss)
        if verbose and batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAcc: {:.6f}'.format(
                epoch, batch_idx * len(batch), len(loader.dataset), 100. * batch_idx / len(loader), loss, train_acc))
    return global_epoch_loss / len(loader.dataset)


def test(loader, model, cuda, verbose=True, data_set='Test', save=None):
    model.eval()
    correct, test_loss = 0., 0.

    indices = list()
    scorez = list()
    targetz = list()
    for keys, data, targets in loader:
        criterion = nn.CrossEntropyLoss(size_average=False)
        if cuda:
            data, targets = data.cuda(), targets.cuda()
            criterion = criterion.cuda()
        data, targets = Variable(data, volatile=True), Variable(targets)
        scores = model(data)
        test_loss += float(criterion(scores, targets))  # sum up batch loss        
        predictions = torch.max(scores, 1)[1].view(targets.size())
        
        correct += (predictions.data == targets.data).cpu().sum()
                
        if save is not None:
            scores = scores.data.cpu().numpy()
            for i, key in enumerate(keys):
                indices.append(int(key))
                scorez.append(scores[i,:])
                targetz.append(int(targets[i]))

    if save is not None:
        np.savez(save, segment_indices=np.asarray(indices), scores=np.asarray(scorez), labels=np.asarray(targetz))

    num_targets = len(loader.dataset)
    test_loss /= num_targets
    accuracy = float(correct) / num_targets
    if verbose:
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
            data_set, test_loss, correct, num_targets, 100 * accuracy))
    return test_loss, accuracy


def main(conf, test_split_name='test'):
    splits = pickle.load(open(conf.splits, "rb"))
    classes_map = pickle.load(open(conf.classes_map, "rb"))
    num_classes = count_num_classes(classes_map)

    conf.cuda = conf.cuda and torch.cuda.is_available()
    torch.manual_seed(conf.seed)
    if conf.cuda:
        torch.cuda.manual_seed(conf.seed)
        print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    composed = transforms.Compose([ToFloatTensor()])
    print(conf)
    if conf.train:
        train_dataset = EpicAudioDataset(conf.audio_hdf5, classes_map, splits,
                                         transform=composed, split_name='train', classification_type=conf.type)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, pin_memory=conf.cuda, sampler=None)

        valid_dataset = EpicAudioDataset(conf.audio_hdf5, classes_map, splits,
                                         transform=composed, split_name='validation', classification_type=conf.type)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, pin_memory=conf.cuda, sampler=None)
        # build model
        if not conf.weights_file:
            if conf.arc.startswith("VGG"):
                model = VGG(conf.arc, num_classes=num_classes)
            elif conf.arc == "TRADDIL" or conf.arc == "TRADDIL16":
                model = TraditionalDilated(num_classes=num_classes)
            else:
                exit(1)
        else:
            model = torch.load(conf.weights_file)

        if conf.cuda:
            model = torch.nn.DataParallel(model).cuda()

        # define optimizer
        if conf.optimizer.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=conf.lr)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=conf.lr, momentum=conf.momentum)

        best_valid_acc = 0
        iteration = 0
        epoch = 1

        # training with early stopping
        while (epoch < conf.epochs + 1) and (iteration < conf.patience):
            train(train_loader, model, optimizer, epoch, conf.cuda,
                  conf.log_interval)  # , weight=train_loader.weight)
            valid_loss, valid_acc = test(
                valid_loader, model, conf.cuda, data_set='Validation')

            if valid_acc <= best_valid_acc:
                iteration += 1
                print('Accuracy was not improved, iteration {0}'.format(
                    str(iteration)))
            else:
                print('Saving state')
                iteration = 0
                best_valid_acc = valid_acc
                state = {
                    'valid_acc': valid_acc,
                    'valid_loss': valid_loss,
                    'epoch': epoch,
                }
                if not os.path.isdir(conf.checkpoint):
                    os.mkdir(conf.checkpoint)
                torch.save(state, './{}/{}_{}_{}_lr_{}_ckpt.t7'.format(
                    conf.checkpoint, conf.snapshot_pref, conf.type, conf.arc, conf.lr))
                torch.save(model.module if conf.cuda else model,
                           './{}/{}_{}_{}_lr_{}_model_{:03d}.t7'.format(conf.checkpoint, conf.snapshot_pref, conf.type,
                                                                        conf.arc, conf.lr, epoch))
            epoch += 1
        del model
    test_dataset = EpicAudioDataset(conf.audio_hdf5, classes_map, splits,
                                    transform=composed, split_name=test_split_name, classification_type=conf.type)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=conf.batch_size, shuffle=None, num_workers=conf.num_workers, pin_memory=conf.cuda, sampler=None)
    state = torch.load('./{}/{}_{}_{}_lr_{}_ckpt.t7'.format(
        conf.checkpoint, conf.snapshot_pref, conf.type, conf.arc, conf.lr))
    if not conf.train and conf.test_epoch:
        epoch = conf.test_epoch
    else:
        epoch = state['epoch']

    print("Testing audio model {} {} {} (epoch {})".format(
        conf.type, conf.arc, conf.lr, epoch))
    model = torch.load('./{}/{}_{}_{}_lr_{}_model_{:03d}.t7'.format(
        conf.checkpoint, conf.snapshot_pref, conf.type, conf.arc, conf.lr, epoch))
    if conf.cuda:
        model = torch.nn.DataParallel(model).cuda()
    results = './{}/{}_{}_{}_lr_{}_model_{:03d}.npz'.format(
        conf.checkpoint, conf.snapshot_pref, conf.type, conf.arc, conf.lr, epoch)
    if test_split_name != 'test':
        results = results.replace('.npz', '_{}.npz'.format(test_split_name))

    print("Saving results in {}".format(results))
    test(test_loader, model, conf.cuda, save=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=lambda p: str(Path(p).absolute()),
                        help='JSON configuration filepath')
    parser.add_argument('-split_name', type=str,
                        default='test', help='Split to evaluate')

    args = parser.parse_args()
    with open(args.conf) as json_file:
        conf = adict(json.load(json_file))

    main(conf, args.split_name)
