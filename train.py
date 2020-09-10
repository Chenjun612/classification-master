import torch.nn as nn
import argparse
import os
import random
import shutil
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
from read_data import CovidDataSet
from k_fold import k_fold_pre
from densenet121 import densenet121
from sklearn.metrics import roc_auc_score
import math
import copy
# used for logging to TensorBoard
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ResNet Training')
parser.add_argument('--model', default='ResNet', type=str,
                    help='baseline of the model')
parser.add_argument('--fold', default=0, type=int,  # 5-fold
                    help='index of k-fold')
parser.add_argument('--n_epoch', default=10, type=int,
                    help='number of epoch to change')
parser.add_argument('--epochs', default=10, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--num_classes', default=1, type=int,  # num_classes 自己加的
                    help='numbers of classes (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,  # 0.01
                    help='initial learning rate')
parser.add_argument('--optimizer', default='SGD', type=str,  # SGD
                    help='optimizer (SGD)')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,  # 正则化参数
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--growth', default=32, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--weight-decay-fc', '--wdfc', default=0, type=float,  # 全连接层正则化参数
                    help='weight decay fc (default: 1e-4)')
parser.add_argument('--seed', default=2, type=int,  # 随机数种子
                    help='random seed(default: 1)')
parser.add_argument('--resume',
                    default='./runs/resnet_COVID_b64_lr0.01_epoch10_225_225/checkpoint/',
                    type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name',
                    default='resnet_COVID_b64_lr0.01_epoch10_225_225',
                    type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard', default=True,
                    help='Log progress to TensorBoard', action='store_true')
parser.add_argument('--use_cuda', default=True,
                    help='whether to use_cuda(default: True)')

DATA_DIR = './data/'
DATA_IMAGE_LIST = './label/data_225covid_225noncovid.txt'
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    global use_cuda, args, writer
    args = parser.parse_args()  # #####很重要
    if args.tensorboard:
        writer = SummaryWriter("runs/%s" % args.name)
    use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.seed > 0:
        seed_torch(args.seed)  # 固定随机数种子
    # create model
    model = densenet121(num_classes=args.num_classes)
    if use_cuda:
        model = model.cuda()
        # for training on multiple GPUs.
        # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
        # model = torch.nn.DataParallel(model).cuda()

    # if args.tensorboard:
    #     writer.add_graph(model, (input_random, input_random, input_random), True)
    if os.path.exists("runs/%s/checkpoint_init.pth.tar" % args.name):
        checkpoint = torch.load("runs/%s/checkpoint_init.pth.tar" % args.name)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        torch.save({'state_dict': model.state_dict()}, "runs/%s/checkpoint_init.pth.tar" % args.name)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    # 5-fold 数据准备
    train_names, val_names = k_fold_pre(filename="runs/%s/data_fold.txt" % args.name, image_list_file=DATA_IMAGE_LIST,
                                        fold=5)

    filelossacc_name = "runs/{}/train_fold{}_loss_acc.txt".format(args.name, args.fold)
    filelossacc = open(filelossacc_name, 'a')
    best_prec = 0  # 第k个fold的准确率
    # 读取第k个fold的数据
    kwargs = {'num_workers': 8, 'pin_memory': True}
    # normalize = transforms.Normalize(mean=[0.465, 0.465, 0.465], std=[0.367, 0.367, 0.367])
    train_datasets = CovidDataSet(data_dir=DATA_DIR, image_list_file=DATA_IMAGE_LIST, fold=train_names[args.fold],
                                  transform=transforms.Compose([transforms.Resize(224),
                                                                transforms.CenterCrop(224),
                                                                transforms.ToTensor()]))
    train_loader = torch.utils.data.DataLoader(dataset=train_datasets, batch_size=args.batch_size, shuffle=True,
                                               **kwargs)

    val_datasets = CovidDataSet(data_dir=DATA_DIR, image_list_file=DATA_IMAGE_LIST, fold=val_names[args.fold],
                                transform=transforms.Compose([transforms.Resize(224),
                                                              transforms.CenterCrop(224),
                                                              transforms.ToTensor()]))
    val_loader = torch.utils.data.DataLoader(dataset=val_datasets, batch_size=args.batch_size, shuffle=True,
                                             **kwargs)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume + 'checkpoint' + str(args.fold) + '.pth.tar'):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume + 'checkpoint' + str(args.fold) + '.pth.tar')
            pretrained_dict = checkpoint['state_dict']
            # pretrained_dict.pop("classifier.weight")
            # pretrained_dict.pop("classifier.bias")
            model.load_state_dict(pretrained_dict)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            print("=> use initial checkpoint")
            checkpoint = torch.load("runs/%s/checkpoint_init.pth.tar" % args.name)
            model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        return 0
    # define loss function
    criterion = nn.BCELoss(reduction='mean')
    # define optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    nesterov=True, weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.99))
    else:
        print('Please choose true optimizer.')
        return 0

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_losses, train_acc = train(train_loader, model, criterion, optimizer, epoch, args.fold)
        # for name, layer in model.named_parameters():
        #     writer.add_histogram('fold' + str(args.fold) + '/' + name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        #     writer.add_histogram('fold' + str(args.fold) + '/' + name + '_data', layer.cpu().data.numpy(), epoch)
        # evaluate on validation set
        val_losses, val_acc, prec1, output_val, label_val, AUROC = validate(val_loader, model, criterion, epoch, args.fold)

        if args.tensorboard:
            # x = model.conv1.weight.data
            # x = vutils.make_grid(x, normalize=True, scale_each=True)
            # writer.add_image('data' + str(k) + '/weight0', x, epoch)  # Tensor
            writer.add_scalars('data' + str(args.fold) + '/loss',
                               {'train_loss': train_losses.avg, 'val_loss': val_losses.avg}, epoch)
            writer.add_scalars('data' + str(args.fold) + '/Accuracy', {'train_acc': train_acc.avg, 'val_acc': val_acc.avg},
                               epoch)
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec
        if is_best == 1:
            best_prec = max(prec1, best_prec)  # 这个fold的最高准确率
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec,
        }, is_best, epoch, args.fold)

        out_write = str(train_losses.avg) + ' ' + str(train_acc.avg) + ' ' + str(val_acc.avg) + '\n'
        filelossacc.write(out_write)
        writer.close()
    filelossacc.write('\n')
    filelossacc.close()


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    # epoch = epoch - 150
    if epoch <= args.n_epoch:
        # lr = args.lr * epoch / args.n_epoch
        lr = args.lr
    else:
        # lr = args.lr
        # lr = args.lr * (0.1 ** (epoch // 50))
        # lr = args.lr * 0.1 * (0.1 ** (epoch // 50))
        # lr = args.lr * (math.e ** (-epoch / args.n_epoch))
        lr = args.lr * (1 + np.cos((epoch - args.n_epoch) * math.pi / args.epochs)) / 2
        # if lr <= 0.000001:
        #     lr = 0.000001
    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('learning_rate', lr, epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch, fold):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    train_losses = AverageMeter()
    train_acc = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        if use_cuda:
            target = target.unsqueeze(1).type(torch.FloatTensor).cuda()
            input = input.type(torch.FloatTensor).cuda()
        output, _ = model(input)

        # measure accuracy and record loss
        train_loss = criterion(output, target)

        train_losses.update(train_loss.item(), input.size(0))
        acc = accuracy(output.data, target)
        train_acc.update(acc, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i == 0 or (i + 1) % args.print_freq == 0 or i == len(train_loader) - 1:  # 按一定的打印频率输出
            print('Train_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.4f}({top1.avg:.4f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          loss=train_losses, top1=train_acc))

    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('data' + str(fold) + '/train_loss', train_losses.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/train_acc', train_acc.avg, epoch)
    return train_losses, train_acc


def validate(val_loader, model, criterion, epoch, fold):  # 返回值为准确率
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    val_losses = AverageMeter()
    val_acc = AverageMeter()
    # switch to evaluate mode  切换到评估模式
    model.eval()  # 很重要
    target_roc = torch.zeros((0, args.num_classes))
    output_roc = torch.zeros((0, args.num_classes))
    end = time.time()
    for i, (input, target, _) in enumerate(val_loader):
        if use_cuda:
            target = target.unsqueeze(1).type(torch.FloatTensor).cuda()
            input = input.type(torch.FloatTensor).cuda()

        # compute output
        output, _ = model(input)

        # measure accuracy and record loss
        val_loss = criterion(output, target)
        print('output:', output.view(-1))
        print('target:', target.view(-1))
        val_losses.update(val_loss.item(), input.size(0))

        target_roc = torch.cat((target_roc, target.data.cpu()), dim=0)
        output_roc = torch.cat((output_roc, output.data.cpu()), dim=0)

        # -------------------------------------Accuracy--------------------------------- #
        acc = accuracy(output.data, target)  # 一个batchsize中n类的平均准确率  输出为numpy类型
        val_acc.update(acc, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i == 0 or (i + 1) % args.print_freq == 0 or i == len(val_loader):  # 按一定的打印频率输出
            print('Val_Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accuracy_avg:.4f}({top1.avg:.4f})'.
                  format(epoch, i, len(val_loader), batch_time=batch_time,
                         loss=val_losses, accuracy_avg=acc, top1=val_acc))

    # -------------------------------------AUROC------------------------------------ #
    AUROC = aucrocs(output_roc, target_roc)
    print('The AUROC is %.4f' % AUROC)
    # -------------------------------------AUROC------------------------------------ #

    # log to TensorBoard
    if args.tensorboard:
        writer.add_scalar('data' + str(fold) + '/val_loss', val_losses.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/val_acc', val_acc.avg, epoch)
        writer.add_scalar('data' + str(fold) + '/val_AUC', AUROC, epoch)

    return val_losses, val_acc, val_acc.avg, output_roc, target_roc, AUROC


def accuracy(output, target):
    output_np = output.cpu().numpy()
    output_np[output_np > 0.5] = 1
    output_np[output_np <= 0.5] = 0

    target_np = target.cpu().numpy()

    right = (output_np == target_np)
    acc = np.sum(right) / output.shape[0]

    return acc


def aucrocs(output, target):  # 改准确度的计算方式

    """
    Returns:
    List of AUROCs of all classes.
    """
    output_np = output.cpu().numpy()
    # print('output_np:',output_np)
    target_np = target.cpu().numpy()
    # print('target_np:',target_np)
    AUROCs=roc_auc_score(target_np[:, 0], output_np[:, 0])
    return AUROCs


def save_checkpoint(state, is_best, epoch, fold):
    """Saves checkpoint to disk"""
    # filename = 'checkpoint' + str(fold) + '_' + str(epoch) + '.pth.tar'
    filename = 'checkpoint' + str(fold) + '.pth.tar'
    directory = "runs/%s/checkpoint/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        torch.save(state, filename)
        shutil.copyfile(filename, 'runs/%s/checkpoint/' % (args.name) + 'model_best' + str(fold) + '.pth.tar')


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


if __name__ == '__main__':

    main()

