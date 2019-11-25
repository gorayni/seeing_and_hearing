from __future__ import print_function, division

import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.realpath('tsn-pytorch'))

from transforms import IdentityTransform
from transforms import ToTorchFormatTensor
from transforms import Stack
from transforms import GroupScale
from transforms import GroupNormalize
from transforms import GroupCenterCrop
from models import TSN
from IO.dataset import EpicTSNDataset
from IO import count_num_classes
import argparse
import json
import numpy as np
import pathlib
import pickle
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torchvision
from addict import Dict as adict
from epic_kitchens.dataset.epic_dataset import EpicVideoDataset
from epic_kitchens.dataset.epic_dataset import EpicVideoFlowDataset
from pathlib import Path
from torch.nn.utils import clip_grad_norm


def train(train_loader, model, criterion, optimizer, epoch, conf):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if conf.no_partialbn:
        model.module.partialBN(False)
    else:
        model.module.partialBN(True)
    # switch to train mode
    model.train()

    end = time.time()
    for i, (_, input_, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input_)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input_.size(0))
        top1.update(prec1, input_.size(0))
        top5.update(prec5, input_.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.backward()

        if conf.clip_gradient is not None:
            total_norm = clip_grad_norm(model.parameters(), conf.clip_gradient)
            if total_norm > conf.clip_gradient:
                print("clipping gradient: {} with coef {}".format(
                    total_norm, conf.clip_gradient / total_norm))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % conf.print_freq == 0:
            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))


def validate(val_loader, model, criterion, conf):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (_, input_, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input_, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data, input_.size(0))
        top1.update(prec1, input_.size(0))
        top5.update(prec5, input_.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % conf.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg


def save_checkpoint(state, is_best, conf, filename='checkpoint.pth.tar'):
    filename = '_'.join((conf.snapshot_pref, conf.arch, conf.class_type, 'lr:{}'.format(conf.lr), conf.modality.lower(),
                         'epoch-' + str(state['epoch']), filename))

    filename = os.path.join(conf.checkpoint, filename)
    torch.save(state, filename)
    if is_best:
        best_name = '_'.join(
            (conf.snapshot_pref, conf.arch, conf.class_type, 'lr:{}'.format(conf.lr), conf.modality.lower(), 'model_best.pth.tar'))
        best_name = os.path.join(conf.checkpoint, best_name)
        shutil.copyfile(filename, best_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = conf.lr * decay
    decay = conf.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def main(conf, best_prec1=0):
    gulp_path = os.path.realpath(conf.gulp_dir)
    gulp_path = Path(gulp_path)
    splits = pickle.load(open(conf.splits, "rb"))
    classes_map = pickle.load(open(conf.classes_map, "rb"))
    num_classes = count_num_classes(classes_map)

    model = TSN(num_classes, conf.num_segments, conf.modality, base_model=conf.arch,
                consensus_type=conf.consensus_type, dropout=conf.dropout, partial_bn=not conf.no_partialbn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=conf.gpus).cuda()

    print(conf)
    if conf.resume:
        if os.path.isfile(conf.resume):
            print(("=> loading checkpoint '{}'".format(conf.resume)))
            checkpoint = torch.load(conf.resume)
            conf.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(conf.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(conf.resume)))

    cudnn.benchmark = True

    # Data loading code
    if conf.modality != 'RGBDiff':
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()

    if conf.modality == 'RGB':
        data_length = 1
    elif conf.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    class_type = 'verb+noun' if conf.class_type == 'action' else conf.class_type
    if conf.modality == 'Flow':
        dataset = EpicVideoFlowDataset(gulp_path=gulp_path, class_type=class_type)
    else:
        dataset = EpicVideoDataset(gulp_path=gulp_path, class_type=class_type)

    train_loader = torch.utils.data.DataLoader(
        EpicTSNDataset(dataset, classes_map, splits, num_segments=conf.num_segments, new_length=data_length,
                       modality=conf.modality, transform=torchvision.transforms.Compose(
                [train_augmentation, Stack(roll=conf.arch == 'BNInception'),
                 ToTorchFormatTensor(div=conf.arch != 'BNInception'), normalize]), split_name='train',
                       classification_type=conf.class_type), batch_size=conf.batch_size, shuffle=True,
        num_workers=conf.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        EpicTSNDataset(dataset, classes_map, splits, num_segments=conf.num_segments, new_length=data_length,
                       modality=conf.modality, random_shift=False, transform=torchvision.transforms.Compose(
                [GroupScale(int(scale_size)), GroupCenterCrop(crop_size), Stack(roll=conf.arch == 'BNInception'),
                 ToTorchFormatTensor(div=conf.arch != 'BNInception'), normalize, ]), split_name='validation',
                       classification_type=conf.class_type), batch_size=conf.val_batch_size, shuffle=False,
        num_workers=conf.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if conf.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(group['name'], len(group['params']),
                                                                             group['lr_mult'], group['decay_mult'])))

    optimizer = torch.optim.SGD(
        policies, conf.lr, momentum=conf.momentum, weight_decay=conf.weight_decay)

    if conf.evaluate:
        validate(val_loader, model, criterion, conf)
        return

    for epoch in range(conf.start_epoch, conf.epochs):
        adjust_learning_rate(optimizer, epoch, conf.lr_steps)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, conf)

        # evaluate on validation set
        if (epoch + 1) % conf.eval_freq == 0 or epoch == conf.epochs - 1:
            prec1 = validate(val_loader, model, criterion, conf)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint(
                {'epoch': epoch + 1, 'arch': conf.arch,
                    'state_dict': model.state_dict(), 'best_prec1': best_prec1, },
                is_best, conf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=lambda p: pathlib.Path(p).absolute(),
                        help='JSON configuration filepath')

    json_conf = parser.parse_args().conf
    with open(json_conf) as json_file:
        conf = adict(json.load(json_file))

    main(conf)
