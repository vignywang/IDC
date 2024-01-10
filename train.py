import os
import argparse
import torch
import torch.nn as nn
from Model import *
from DataLoader import ImageDataset
from torch.autograd import Variable
from utils.accuracy import *
from utils.lr import *
from utils.optimizer import *
import os
import random
import time
import torch.nn.functional as F
import numpy as np
seed = 0 #6
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        ##  path
        self.parser.add_argument('--root', type=str, default='CUB_200_2011')
        self.parser.add_argument('--test_gt_path', type=str, default='CUB_200_2011/test_bounding_box.txt')
        self.parser.add_argument('--num_classes', type=int, default=200)
        self.parser.add_argument('--test_txt_path', type=str, default='CUB_200_2011/test_list.txt')
        ##  save
        self.parser.add_argument('--save_path', type=str, default='logs')
        self.parser.add_argument('--load_path', type=str, default='VGG.pth.tar')
        ##  dataloader
        self.parser.add_argument('--crop_size', type=int, default=224)
        self.parser.add_argument('--resize_size', type=int, default=256)
        self.parser.add_argument('--num_workers', type=int, default=1)
        self.parser.add_argument('--nest', action='store_true')
        ##  train
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--epochs', type=int, default=100)
        self.parser.add_argument('--phase', type=str, default='train')  ## train / test
        self.parser.add_argument('--lr', type=float, default=0.001)
        self.parser.add_argument('--weight_decay', type=float, default=5e-4)
        self.parser.add_argument('--power', type=float, default=0.9)
        self.parser.add_argument('--momentum', type=float, default=0.9)
        self.parser.add_argument('--b', type=float, default=0.5)
        self.parser.add_argument('--c', type=float, default=1)
        self.parser.add_argument('--d', type=float, default=0.25)
        self.parser.add_argument('--e', type=float, default=2)
        ##  model
        self.parser.add_argument('--arch', type=str,
                                 default='vgg')  ##  choose  [ vgg, resnet, inception, mobilenet ]
        ##  show
        self.parser.add_argument('--show_step', type=int, default=118)
        ##  GPU'
        self.parser.add_argument('--gpu', type=str, default='0')

    def parse(self):
        opt = self.parser.parse_args()
        opt.arch = opt.arch
        return opt


args = opts().parse()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

lr = args.lr

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y

def gen_grid(h_min, h_max, w_min, w_max, len_h, len_w):
    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w), torch.linspace(h_min, h_max, len_h)])
    grid = torch.stack((x, y), -1).transpose(0, 1).reshape(-1, 2).float().cuda()
    return grid

def get_expected_correspondence_loc(map):
    B, d, h, w = map.size()
    grid_n = gen_grid(-1, 1, -1, 1, h, w)
    flaten=map.view(B,d,-1)
    prob = F.softmax(flaten,dim=-1)
    expected_coord_n = torch.sum(grid_n * prob.unsqueeze(-1), dim=2)  # Bxnx2
    return expected_coord_n

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    delta = (b - a) / n_bins
    bins = torch.arange(a, b + 1e-8, step=delta)
    count = torch.histc(x, n_bins).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins

class HistoLoss(nn.Module):
    def __init__(self, n_bins=10):
        super().__init__()
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        self.n_bins=n_bins

    def forward(self, x_real, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_real.shape[2]):
            x_i = x_real[..., i].reshape(-1, 1)
            d, b = histogram_torch(x_i, self.n_bins, density=True)
            self.densities.append(nn.Parameter(d).to(x_real.device))
            delta = b[1:2] - b[:1]
            loc = 0.5 * (b[1:] + b[:-1])
            self.locs.append(loc)
            self.deltas.append(delta)

        for i in range(x_fake.shape[2]):
            loc = self.locs[i].view(1, -1).to(x_fake.device)
            x_i = x_fake[:, :, i].contiguous().view(-1, 1).repeat(1, loc.shape[1])
            dist = torch.abs(x_i - loc)
            counter = (relu(self.deltas[i].to(x_fake.device) / 2. - dist) > 0.).float()
            density = counter.mean(0) / self.deltas[i].to(x_fake.device)
            abs_metric = torch.abs(density - self.densities[i].to(x_fake.device))
            loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise

def kurtosis_torch(x, dim=(2, 3), excess=True, dropdims=True):
    x = x - x.mean(dim, keepdims=True)
    x_4 = torch.pow(x, 4).mean(dim, keepdims=True)
    x_var2 = torch.pow(torch.var(x, dim=dim, unbiased=False, keepdims=True), 2)
    kurtosis = x_4 / x_var2
    if excess:
        kurtosis = kurtosis - 3
    if dropdims:
        #kurtosis = kurtosis[0, 0]
        kurtosis = kurtosis.squeeze(-1).squeeze(-1).squeeze(-1)
    return kurtosis

def dis_loss(x,y):
    dis = torch.sqrt(torch.pow(x - y, 2).sum(2))
    dis = torch.mean(dis)
    return dis

def dis_loss2(x,y):
    x_std = x.std((2,3), keepdims=True)
    y_std = y.std((2,3), keepdims=True)
    return torch.mean(torch.abs(1-(x_std/y_std)))

class Cls_loss(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)

if __name__ == '__main__':
    if args.phase == 'train':
        MyData = ImageDataset(args,args.phase)
        MyDataLoader = torch.utils.data.DataLoader(dataset=MyData, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.num_workers)
        ##  model
        model = eval(args.arch).model(args, pretrained=True).cuda()
        model.train()
        ##  optimizer
        optimizer = get_optimizer(model, args)
        loss_func = nn.CrossEntropyLoss().cuda()
        loss_smooth = Cls_loss().cuda()
        l1_loss = torch.nn.L1Loss()
        epoch_loss = 0
        b = args.b
        c = args.c
        d = args.d
        e = args.e
        loss_best=1000
        scale_factor=0.75
        print('Train begining!')
        for epoch in range(0, args.epochs):
            ##  accuracy
            cls_acc_1 = AverageMeter()
            cls_acc_2 = AverageMeter()
            loss_total = AverageMeter()
            loss_epoch_1 = AverageMeter()
            loss_epoch_2 = AverageMeter()
            loss_epoch_3 = AverageMeter()
            loss_epoch_4 = AverageMeter()
            poly_lr_scheduler(optimizer, epoch, decay_epoch=20)
            for step, (path, imgs, label) in enumerate(MyDataLoader):
                imgs, label = Variable(imgs).cuda(), label.cuda()
                ##  backward
                optimizer.zero_grad()
                N, C, H, W = imgs.size()
                img2 = F.interpolate(imgs, scale_factor=scale_factor, mode='bilinear', align_corners=True)
                output1_1, output1_2,cam1,ins_loss1 = model(imgs, label, 1)
                output2_1, output2_2,cam2,ins_loss2 = model(img2, label, 1)
                label = label.long()
                cam1 = F.interpolate(cam1, scale_factor=scale_factor, mode='bilinear',align_corners=True) #scale_factor=scale_factor
                loss1_1 = loss_smooth(output1_1, label).cuda()
                loss1_2 = loss_smooth(output1_2, label).cuda()
                loss_1 = loss1_1
                loss_2 = loss1_2
                loss_3 = ins_loss1
                ns, cs, hs, ws = cam2.size()
                cam1 = F.sigmoid(cam1)
                cam2 = F.sigmoid(cam2)
                loss_4 = torch.abs(cam1-cam2).cuda()
                loss_4 = torch.mean(torch.topk(loss_4.view(ns, -1), k=(int)(hs * ws * 0.2), dim=-1)[0])

                # Instance-level center constraints
                center1 = get_expected_correspondence_loc(cam1)
                center2 = get_expected_correspondence_loc(cam2)
                loss_5 = dis_loss(center1,center2)

                loss = loss_1 + loss_2 * b + loss_3 * c + loss_4*e + loss_5*d
                loss.backward()
                optimizer.step()

                ##  count_accuracy
                cur_batch = label.size(0)
                cur_cls_acc_1 = 100. * compute_cls_acc(output1_1, label)
                cls_acc_1.updata(cur_cls_acc_1, cur_batch)
                cur_cls_acc_2 = 100. * compute_cls_acc(output1_2, label)
                cls_acc_2.updata(cur_cls_acc_2, cur_batch)
                loss_epoch_1.updata(loss_1.data, 1)
                loss_epoch_2.updata(loss_2.data, 1)
                loss_epoch_3.updata(loss_3.data, 1)
                loss_epoch_4.updata(loss_4.data+loss_5.data, 1)
                loss_total.updata(loss.data,1)
                if (step + 1) % args.show_step == 0:
                    print(
                        '  Epoch:[{}/{}]\t step:[{}/{}]\tcls_loss_1:{:.3f}\tcls_loss_2:{:.3f}\tins_loss:{:.3f}\tcam_loss:{:.3f}\t cls_acc_1:{:.2f}%\tcls_acc_2:{:.2f}%'.format(
                            epoch + 1, args.epochs, step + 1, len(MyDataLoader), loss_epoch_1.avg, loss_epoch_2.avg,
                            loss_epoch_3.avg, loss_epoch_4.avg, cls_acc_1.avg, cls_acc_2.avg
                        ))

            if epoch % 1 == 0:
                torch.save(model.state_dict(),
                           os.path.join('logs/' + args.arch + '/' + args.arch + str(epoch) + '.pth.tar'),
                           _use_new_zipfile_serialization=False)
                if loss_best>loss_total.avg:
                    loss_best=loss_total.avg
                    torch.save(model.state_dict(),
                               os.path.join('logs/' + args.arch + '/' + args.arch + 'best.pth.tar'),
                               _use_new_zipfile_serialization=False)
                    print(str(epoch)+"best_loss:"+str(loss_best))





