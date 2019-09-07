# -*- coding:utf-8 -*-
import os
import random
from timeit import default_timer as timer
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import numpy as np
import albumentations as A
import pretrainedmodels
from DataLoader.folder import ImageFolder

class LabelSmoothing(nn.Module):

    def __init__(self, num_classes, smoothing=0.9, use_gpu=True):
        super(LabelSmoothing, self).__init__()
        assert 0 < smoothing < 1., "smoothing value should be between 0 and 1."
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.use_gpu = use_gpu

    def forward(self, x, target):

        batch_size = target.size(0)
        if self.use_gpu:
            smoothed_labels = torch.full(size=(batch_size, self.num_classes),\
                                        fill_value=(1-self.smoothing) / (self.num_classes-1)).cuda()
        smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(target, dim=1), value=self.smoothing)
        log_prob = F.log_softmax(x, dim=1)
        loss = -torch.sum(log_prob*smoothed_labels)/batch_size

        return loss

class ImgAugTransform:
    def __init__(self):
        self.extra_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=1),
        A.Cutout(num_holes=8, max_h_size=20, max_w_size=20),
        ], p=1)

    def __call__(self, img):
        img = np.array(img)
        img = self.extra_transform(image=img)['image']
        return Image.fromarray(img)

def train(opt, train_loader, model, optimizer, scheduler, criterion, epoch):
    model.train()
    scheduler.step()

    running_loss = 0.0
    running_correct = 0.0
    total = 0.0

    since = timer()
    print(f"Epoch {epoch}/{opt.max_epoches}")
    print('-'*10)

    loader = tqdm(train_loader, total=len(train_loader))
    for batch_index, (image, label) in enumerate(loader):
        loader.set_description(f"Iter {batch_index}")

        image = image.cuda()
        label = label.cuda()

        optimizer.zero_grad()

        output = model(image)
        _, pred = torch.max(output, 1)
        
        loss = criterion(output, label)

        # backward + optimizer
        loss.backward()
        optimizer.step()

        lr = [ param_group['lr'] for param_group in optimizer.param_groups ]

        running_loss += loss.item()
        running_correct += (pred == label).sum().item()
        total += label.size(0)

        loader.set_postfix(Epoch=f"{epoch:2}", loss=f"{loss.item():6.5f}", lr=f"{lr}")

    epoch_loss = running_loss / len(train_loader)

    time_elapsed = timer() - since

    print(f"===> Train Phase: loss: {epoch_loss: .4f} Acc: {100*running_correct/total:.3f}% \
           Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

def test(opt, test_loader, model, criterion, epoch):

    model.eval()

    running_loss = 0.0
    running_correct = 0.0
    total = 0.0

    since = timer()
    with torch.no_grad():
        for batch_index, (image, label) in enumerate(test_loader):

            image = image.cuda()
            label = label.cuda()

            output = model(image)
            _, pred = torch.max(output, 1)
            loss = criterion(output, label)

            running_loss += loss.item()
            running_correct += (pred == label).sum().item()
            total += label.size(0)

    epoch_loss = running_loss / len(test_loader)
    time_elapsed = timer() - since

    epoch_acc = round(100*running_correct/total, 3)

    if epoch_acc > opt.Acc:
        opt.Acc = epoch_acc

    print(f"===> Validate Phase: loss: {epoch_loss: .5f} Acc: {epoch_acc:.3f}% Best Acc: {opt.Acc:.3f}% \
            Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s ")
    print()

def setseed(args):

    cudnn.enabled = True
    # cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
     

def setgpu(args):
    # set gpu ids
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    return args.gpu_ids

def main(args):

    setseed(args)

    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        ImgAugTransform(),
        transforms.ToTensor(),
        transforms.Normalize([0.387, 0.396, 0.350], [0.136, 0.129, 0.123])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.387, 0.396, 0.350], [0.136, 0.129, 0.123])
    ])
    }

    traindata = ImageFolder(root=args.train_root, classind=args.classind, transform=data_transforms['train'])
    valdata = ImageFolder(root=args.val_root, classind=args.classind, transform=data_transforms['val'])

    trainloader = DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valloader = DataLoader(valdata, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # define gpus
    gpu_ids = setgpu(args)

    # define model
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, args.num_classes)
    model = model.cuda()
    model = torch.nn.DataParallel(model, gpu_ids)

    # define loss
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothing(num_classes=args.num_classes, smoothing=0.9, use_gpu=True)

    # define optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),\
                          lr=args.lr,
                          weight_decay=args.weight_decay,
                          momentum=0.9)

    # define lr_scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1, last_epoch=-1)

    for epoch in range(1, args.max_epoches+1):

        train(args, trainloader, model, optimizer, scheduler, criterion, epoch)
        test(args, valloader, model, criterion, epoch)


if __name__ == '__main__':

    import timeit
    start = timeit.default_timer()
    import argparse
    parser = argparse.ArgumentParser()
    #++++++++++++++++++ model name +++++++++++++++++++#
    parser.add_argument("--model_name", type=str, default='se_resnet50', help="choose which model to train.")

    #++++++++++++++++++ base hyper parameter ++++++++++++++++++++++#
    parser.add_argument("--num_classes", type=int, default=45, help="RS Dataset have 45 classes")

    #++++++++++++++++++ File Path +++++++++++++++++#
    parser.add_argument('--classind', type=str, default='/home/syz/workspace/RS_competition/RS_dataset/ClsName2id.txt', help=" the path of class_ind ")
    parser.add_argument('--train_root', type=str, default='/home/syz/workspace/RS_competition/RS_dataset/train', help=" the path to video folder ")
    parser.add_argument('--val_root', type=str, default='/home/syz/workspace/RS_competition/RS_dataset/val', help=" the path to video folder ")
    parser.add_argument("--save_dir", type=str, default='./save_dir/', help='path to save_dir!')

    #++++++++++++++++++ datset hyper parameters +++++++++++++++++++#
    parser.add_argument("--max_epoches", type=int, default=30, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=4, help=" ")

    #++++++++++++++++++ optimizer hyper parameters ++++++++++++++++#
    parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
    parser.add_argument("--weight_decay", type=float, default=4e-5, help="weight decay for optimizer.")
    parser.add_argument("--momentum", type=float, default=0.9, help="weight decay for optimizer.")

    #++++++++++++++++++ lr scheduler hyper parameters +++++++++++++#
    parser.add_argument("--step_size", type=int, default=10, help="step size for StepLR scheduler.")

    #++++++++++++++++++ Run Set ++++++++++++++++++++++#
    parser.add_argument("--gpu_ids", type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2.')
    parser.add_argument("--seed", type=int, default=1, help="set random seed.")

    # #++++++++++++++++++Best_Acc+++++++++++++++++++++++#
    # parser.add_argument('--L2sp', type=Ture, default=0.0, help="save the Acc")

    #++++++++++++++++++Best_Acc+++++++++++++++++++++++#
    parser.add_argument('--Acc', type=float, default=0.0, help="save the Acc")



    args = parser.parse_args()
    print('++++++++++++++++++ makesure parameters +++++++++++++++++++++++++++')
    for k, v in vars(args).items():
        print('Hyper Paramters: {:20s} \t Value: {}'.format(k, v))
    print('++++++++++++++++ above are all parameters ++++++++++++++++++++++++')

    main(args)
    end = timeit.default_timer()
    print("training time:", 1.0*(end-start)/3600)
