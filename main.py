import argparse
import json
import logging
import math
import os
from os.path import exists, join, split

import time

import numpy as np
import shutil

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from models.FCN8s import FCN8s

import data_transforms as transforms
from tensorboardX import SummaryWriter
import pdb

try:
    from modules import batchnormsync
except ImportError:
    pass
sys.path.insert(0,'./cityscapesScripts/cityscapesscripts/evaluation')
from city_meanIU import city_meanIU, labels

trainId2Id = { label.trainId : label.id for label in reversed(labels) }
del trainId2Id[255]
del trainId2Id[-1]
trainId2Id[255] = 0
writer = SummaryWriter('runs/synthia_all_scratch')

CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('-d', '--data-dir', default=None, required=True)
    parser.add_argument('-l', '--list-dir', default=None,
                        help='List dir to look for train_images.txt etc. '
                             'It is the same with --data-dir if not set.')
    parser.add_argument('-c', '--classes', default=0, type=int)
    parser.add_argument('-s', '--crop-size', default=0, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step')
    parser.add_argument('--dataset', type=str, default='SYNTHIA')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--phase', default='val')
    parser.add_argument('--random-scale', default=0, type=float)
    parser.add_argument('--random-rotate', default=0, type=int)
    args = parser.parse_args()

    assert args.classes > 0

    print(' '.join(sys.argv))
    print(args)

    return args

FORMAT = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SegDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, phase, transforms=None, list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms
        self.image_list = None
        self.label_list = None
        self.read_lists()

    def __getitem__(self, index):
        data = [Image.open(join(self.data_dir, self.image_list[index]))]
        if self.label_list is not None:
            data.append(Image.open(
                join(self.data_dir, self.label_list[index])))
        assert exists(join(self.data_dir, self.label_list[index]))
        assert exists(join(self.data_dir, self.image_list[index]))

        if self.transforms is not None:
            data = list(self.transforms(*data))
            #data = [self.transforms(im) for im in data]
        if self.out_name:
            #if self.label_list is None:
            #    data.append(data[0][0, :, :])
            data.append(self.image_list[index])
        return tuple(data)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        image_path = join(self.list_dir, self.phase + '_images_resized.txt')
        label_path = join(self.list_dir, self.phase + '_labels_resized.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)

def validate(val_loader, model, dataset, criterion, eval_score=None, eval_score_miou=None, print_freq=10):
    # pdb.set_trace()
    batch_time = AverageMeter()
    losses = AverageMeter()
    score = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    hist = np.zeros((19, 19))
    for i, (input, target) in enumerate(val_loader):
        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()
        input = input.cuda()
        target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0].item(), input.size(0))
        # if eval_score is not None:
        #     score.update(eval_score(output, target_var), input.size(0))
        hist_local, _ = eval_score_miou(output.detach(), target_var.detach(), dataset)
        hist += hist_local

        acc = np.diag(hist_local).sum() / hist_local.sum() * 100
        acc_total = np.diag(hist).sum() / hist.sum() * 100
        mIoU = np.nanmean(per_class_iu(hist_local, dataset)) * 100
        mIoU_total = np.nanmean(per_class_iu(hist, dataset)) * 100

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {acc:.3f} ({acc_total:.3f})\t'
                        'mIoU {mIoU:.3f} ({mIoU_total:.3f})'.format(
                    i, len(val_loader), acc=acc, acc_total=acc_total, batch_time=batch_time, 
                    loss=losses, mIoU=mIoU, mIoU_total=mIoU_total))

    mIoU_total = np.nanmean(per_class_iu(hist, dataset)) * 100
    print(' * Score {mIoU:.3f}'.format(mIoU=mIoU_total))

    return mIoU_total 


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


def save_output_images(predictions, filenames, output_dir):
    # pdb.set_trace()
    for ind in range(len(filenames)):
        im = Image.fromarray(predictions[ind].astype(np.uint8))
        fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
        out_dir = split(fn)[0]
        if not exists(out_dir):
            os.makedirs(out_dir)
        im.save(fn)

def save_colorful_images(predictions, filenames, output_dir, palettes):
   for ind in range(len(filenames)):
       im = Image.fromarray(palettes[predictions[ind].squeeze()])
       fn = os.path.join(output_dir, filenames[ind][:-4] + '.png')
       out_dir = split(fn)[0]
       if not exists(out_dir):
           os.makedirs(out_dir)
       im.save(fn)

def accuracy(output, target):
    # batch_size = target.size(0) * target.size(1) * target.size(2)
    _, pred = output.max(1)
    pred = pred.view(1, -1)
    target = target.view(1, -1)
    correct = pred.eq(target)
    correct = correct[target != 255]
    correct = correct.view(-1)
    score = correct.float().sum(0).mul(100.0 / correct.size(0))
    return score.item()

def accuracy_miou(output, target):
    _, pred = output.max(1)
    target = target.clone().cpu().numpy().astype(np.uint8)
    pred = pred.clone().cpu().numpy().astype(np.uint8)
    for key in trainId2Id.keys():
        target[target == key] = trainId2Id[key]
        pred[pred == key] = trainId2Id[key]
    results_dict=city_meanIU(target, pred)
    val_mean_IU=results_dict['averageScoreClasses']
    pdb.set_trace()
    return val_mean_IU

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist, dataset="SYNTHIA"):
    # return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

    if dataset == "SYNTHIA":
        hist = np.delete(hist, [9, 14, 16], 0)
        hist = np.delete(hist, [9, 14, 16], 1)
    res = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return res

def fast_miou(output, target, dataset, num_classes=19):
    _, pred = torch.max(output, 1)
    pred = pred.cpu().data.numpy()
    target = target.cpu().data.numpy()
    hist = fast_hist(pred.flatten(), target.flatten(), num_classes)
    mIoU = np.nanmean(per_class_iu(hist, dataset)) * 100
    return hist, mIoU


def train(train_loader, model, dataset, criterion, optimizer, epoch,
          eval_score=None, eval_score_miou=None, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    scores_iou = AverageMeter()

    model.train()

    end = time.time()

    frac = 0.0625
    hist = np.zeros((19, 19))
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if type(criterion) in [torch.nn.modules.loss.L1Loss,
                               torch.nn.modules.loss.MSELoss]:
            target = target.float()

        input = input.cuda()
        target = target.cuda(async=True)

        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)#[0]

        counts = np.bincount(target.cpu().numpy().flatten())[:19]
        avg = counts.sum() / np.sum(counts > 0)
        weights = avg / (counts + 0.01) 
        criterion = nn.NLLLoss2d(ignore_index=255, weight=torch.FloatTensor(weights)).cuda()

        loss = criterion(output, target_var)

        writer.add_scalar('data/loss', loss.item(), epoch * len(train_loader)*frac + i)

        losses.update(loss.item(), input.size(0))

        hist_local, _ = eval_score_miou(output.detach(), target_var.detach(), dataset)
        hist += hist_local

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        acc = np.diag(hist_local).sum() / hist_local.sum() * 100

        acc_total = np.diag(hist).sum() / hist.sum() * 100
        mIoU = np.nanmean(per_class_iu(hist_local, dataset)) * 100
        writer.add_scalar('data/mIoU', mIoU, epoch * len(train_loader)*frac + i)
        mIoU_total = np.nanmean(per_class_iu(hist, dataset)) * 100

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Score {acc:.3f} ({acc_total:.3f})\t'
                        'mIoU {mIoU:.3f} ({mIoU_total:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, mIoU=mIoU, mIoU_total=mIoU_total, 
                acc=acc, acc_total=acc_total))
        if i > frac * len(train_loader):
            break


def save_checkpoint(state, is_best_sim, is_best_real, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best_sim:
        shutil.copyfile(filename, './checkpoints/model_best_sim.pth.tar')
    if is_best_real:
        shutil.copyfile(filename, './checkpoints/model_best_real.pth.tar')


def test(eval_data_loader, model, num_classes, output_dir='pred', save_vis=False):
    dataset="GTA"
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    hist = np.zeros((num_classes, num_classes))
    label_all = None
    pred_all = None
    for iter, (image, label, name) in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        with torch.no_grad():
            final = model(image)
            _, pred = torch.max(final, 1)

        label_all = label.cpu() if label_all is None else torch.cat((label_all, label.cpu()), dim=0)
        pred_all  = final.cpu()  if pred_all  is None else torch.cat((pred_all, final.cpu()), dim=0)

        print("=================")
        print(label_all.shape)
        print(pred_all.shape)

        _, miou = fast_miou(pred_all, label_all, 'GTA')
        pred = pred.cpu().data.numpy()
        batch_time.update(time.time() - end)
        name = [split(single_name)[1] for single_name in name]
        if save_vis:
            #save_output_images(pred, name, output_dir)
            save_colorful_images(
                pred, name, output_dir + '_color',
                TRIPLET_PALETTE if num_classes == 3 else CITYSCAPE_PALETTE)
        label = label.numpy()
        hist += fast_hist(pred.flatten(), label.flatten(), num_classes)
        print('===> mAP {mAP:.3f}'.format(
            mAP=round(np.nanmean(per_class_iu(hist, dataset)) * 100, 2)))

        end = time.time()
        print('Eval: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(iter, len(eval_data_loader), batch_time=batch_time,
                            data_time=data_time))
    ious = per_class_iu(hist, dataset) * 100
    print(' '.join('{:.03f}'.format(i) for i in ious))
    return round(np.nanmean(ious), 2)


def test_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    phase = args.phase

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = FCN8s(args.classes, pretrained=True)
    model = torch.nn.DataParallel(single_model).cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            best_prec1_real = checkpoint['best_prec1_real']
            best_prec1_sim = checkpoint['best_prec1_sim']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    out_dir = './result_imgs/{:03d}_{}'.format(start_epoch, phase)

    data_dir = args.data_dir
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])

    test_loader = torch.utils.data.DataLoader(
        SegDataset(data_dir=data_dir, phase=phase, transforms=transforms.Compose([transforms.ToTensor(), normalize]), out_name=True),
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True
    )

    cudnn.benchmark = True
    mAP = test(test_loader, model, args.classes, save_vis=True, output_dir=out_dir)
    print('mAP: %f', mAP)


def train_seg(args):
    batch_size = args.batch_size
    num_workers = args.workers
    crop_size = args.crop_size
    dataset = args.dataset

    print(' '.join(sys.argv))

    for k, v in args.__dict__.items():
        print(k, ':', v)

    single_model = FCN8s(args.classes, pretrained=True)

    model = torch.nn.DataParallel(single_model).cuda()
    criterion = nn.NLLLoss2d(ignore_index=255)

    criterion.cuda()

    data_dir = args.data_dir
    info = json.load(open(join('./data/' + dataset, 'info.json'), 'r'))
    normalize_SYNTHIA = transforms.Normalize(mean=info['mean'],
                                             std=info['std'])
    info = json.load(open(join('./data/cityscapes/', 'info.json'), 'r'))
    normalize_cityscapes = transforms.Normalize(mean=info['mean'],
                                             std=info['std'])

    t_SYNTHIA = []
    crop_size_SYNTHIA = (1280, 640)
    scale_SYNTHIA = (720.0/1280.0, 720./1280.)
    t_SYNTHIA.extend([transforms.ToTensor(),
                      normalize_SYNTHIA])

    t_cityscapes = []
    crop_size_cityscapes = (0, 0)
    scale_cityscapes = (720./2048., 720./2048.)
    t_cityscapes.extend([transforms.ToTensor(),
                         normalize_cityscapes])

    train_loader = torch.utils.data.DataLoader(
        SegDataset(data_dir='./data/' + dataset, phase='train', transforms=transforms.Compose(t_SYNTHIA), out_name=False),
        batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )

    val_loader_sim = torch.utils.data.DataLoader(
        SegDataset(data_dir='./data/' + dataset, phase='val', transforms=transforms.Compose(t_SYNTHIA), out_name=False),
        batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    val_loader_real = torch.utils.data.DataLoader(
        SegDataset(data_dir='./data/cityscapes/', phase='val', transforms=transforms.Compose(t_cityscapes), out_name=False),
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True
    )
    test_loader_real = torch.utils.data.DataLoader(
        SegDataset(data_dir='./data/cityscapes/', phase='test', transforms=transforms.Compose(t_cityscapes), out_name=False),
        batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    cudnn.benchmark = True
    best_prec1_sim = 0
    best_prec1_real = 0
    start_epoch = 0

    # optionally resume from a checkpoint
    if args.resume:
       if os.path.isfile(args.resume):
           print("=> loading checkpoint '{}'".format(args.resume))
           checkpoint = torch.load(args.resume)
           model.load_state_dict(checkpoint['state_dict'])
           optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
           print("=> loaded checkpoint '{}' (epoch {})"
                 .format(args.resume, checkpoint['epoch']))
       else:
           print("=> no checkpoint found at '{}'".format(args.resume))

    #validate(val_loader_real, model, dataset, criterion, eval_score=accuracy, eval_score_miou=fast_miou)       
    #validate(test_loader_real, model, dataset, criterion, eval_score=accuracy, eval_score_miou=fast_miou)
    # prec1_sim = validate(val_loader_sim, model, dataset, criterion, eval_score=accuracy, eval_score_miou=fast_miou)
    for epoch in range(start_epoch, args.epochs):
        print("=====================================")
        print('Epoch: [{0}]'.format(epoch))

        train(train_loader, model, dataset, criterion, optimizer, epoch,
              eval_score=accuracy, eval_score_miou=fast_miou)

        if (epoch + 1) % 5 == 0:
            prec1_sim = validate(val_loader_sim, model, dataset, criterion, eval_score=accuracy, eval_score_miou=fast_miou)
        else:
            prec1_sim = 0
        is_best_sim = prec1_sim > best_prec1_sim
        best_prec1_sim = max(prec1_sim, best_prec1_sim)

        prec1_real = validate(val_loader_real, model, dataset, criterion, eval_score=accuracy, eval_score_miou=fast_miou)
        is_best_real = prec1_real > best_prec1_real
        best_prec1_real = max(prec1_real, best_prec1_real)

        writer.add_scalar('data/prec1_real', prec1_real, epoch)

        checkpoint_path = './checkpoints/checkpoint_latest.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_prec1_real': best_prec1_real,
            'best_prec1_sim': best_prec1_sim
        }, is_best_sim=is_best_sim, is_best_real=is_best_real, filename=checkpoint_path)

        if (epoch + 1) % 20 == 0:
            history_path = './checkpoints/checkpoint_{:03d}.pth.tar'.format(epoch + 1)
            shutil.copyfile(checkpoint_path, history_path)

def main():
    args = parse_args()
    if args.cmd == 'train':
        train_seg(args)
    elif args.cmd == 'test':
        test_seg(args)
        pass


if __name__ == '__main__':
    main()
