import os
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
from torch.autograd import Variable
from utils.util import *
import torch.nn.functional as F
import numpy as np
import cv2
import copy
import matplotlib.pyplot as plt
from utils.loss import *

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 base_width=64):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def map_process(pros,map,T=2):
    weight = F.softmax(pros / T, dim=-1)
    fmap = map * weight.unsqueeze(-1).unsqueeze(-1)
    fmap = torch.sum(fmap, dim=1, keepdim=True)
    return fmap

#ResNet-50 Backbone
class IDC(nn.Module):
    def __init__(self, block, layers, config, large_feature_map=True):
        super(IDC, self).__init__()
        self.config=config
        stride_l3 = 1 if large_feature_map else 2
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier_cls = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 200, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )

        self.classifier_loc = nn.Sequential(
            nn.Conv2d(1024 + 512, 200, kernel_size=3, padding=1),
            #nn.Conv2d(512, 200, kernel_size=3, padding=1),
        )
        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, label=None):
        classifier_cls_copy = copy.deepcopy(self.classifier_cls)
        pool3_copy=copy.deepcopy(self.pool3)
        layer3_copy = copy.deepcopy(self.layer3)
        pool4_copy = copy.deepcopy(self.pool4)
        layer4_copy = copy.deepcopy(self.layer4)
        self.batch = x.size(0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_2 = x.clone()
        x = self.pool3(x)
        x = self.layer3(x)
        x_3 = x.clone()
        x = self.pool4(x)
        x = self.layer4(x)
        x = self.classifier_cls(x)
        self.prob1 = self.avg_pool(x).squeeze(-1).squeeze(-1)
        p_label = label.unsqueeze(-1)
        self.fmap,self.center = self.tfm_modeling(x_3, x_2, p_label)
        self.bmap = 1 - self.fmap
        x_erase1 = x_2.detach() * (self.fmap)
        x_erase2 = x_2.detach() * (self.bmap)
        x_erase = torch.cat((x_erase1, x_erase2), dim=0)
        x_erase = pool3_copy(x_erase)
        x_erase = layer3_copy(x_erase)
        x_erase = pool4_copy(x_erase)
        x_erase = layer4_copy(x_erase)
        x_erase = classifier_cls_copy(x_erase)
        x_erase = self.avg_pool(x_erase).squeeze(-1).squeeze(-1)
        self.prob2, self.prob3 = torch.chunk(x_erase, 2, dim=0)
        tfm_loss = self.TFM_loss()
        return self.prob1, self.prob2 , self.fmap, tfm_loss,self.center

    def evl_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_2 = x.clone()
        x = self.maxpool(x)
        x = self.layer3(x)
        x_3 = x.clone()
        x = F.max_pool2d(x, kernel_size=2)
        x = self.layer4(x)
        x = self.classifier_cls(x)
        self.prob1 = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x_3 = F.interpolate(x_3, x_2.shape[2:], mode='bilinear')
        fmaps_all = self.classifier_loc(torch.cat((x_2, x_3), dim=1))
        fmaps_all = torch.sigmoid(fmaps_all)
        self.fmap = map_process(self.prob1,fmaps_all,self.config['tua'])

        return self.prob1

    def tfm_modeling(self,x_3,x_2,p_label):
        x_3 = F.interpolate(x_3, x_2.shape[2:], mode='bilinear')
        fmap_all = self.classifier_loc(torch.cat((x_2, x_3), dim=1))
        
        #fmap_all = self.classifier_loc(x_2)
        fmap = torch.zeros(self.batch, 1, x_2.size(-2), x_2.size(-1)).cuda()
        for i in range(self.batch):
            fmap[i][0] = fmap_all[i][p_label[i]].mean(0)
        center = self.get_expected_correspondence_loc(fmap)
        return F.sigmoid(fmap),center

    def TFM_loss(self):
        dis_neg = cos_distance(self.prob1, self.prob3)
        neg_pair = torch.diagonal(dis_neg, dim1=0, dim2=1)
        dis_pos = cos_distance(self.prob1, self.prob2)
        pos_pair = torch.diagonal(dis_pos, dim1=0, dim2=1)
        triple_loss = torch.relu(1 + pos_pair - neg_pair)
        res = triple_loss
        forepixel = self.fmap
        forepixel = forepixel.clone().view(self.batch, -1)
        forepixel = forepixel.mean(1)
        loss = res + forepixel
        loss = loss.mean(0)

        return loss

    def get_expected_correspondence_loc(self, atm):
        B, d, h, w = atm.size()
        grid_n = self.gen_grid(-1, 1, -1, 1, h, w)
        flaten = atm.view(B, d, -1)
        prob = F.softmax(flaten, dim=-1)
        expected_coord_n = torch.sum(grid_n * prob.unsqueeze(-1), dim=2)  # Bxnx2
        return expected_coord_n

    def _make_layer(self, block, planes, blocks, stride):
        layers = self._layer(block, planes, blocks, stride)
        return nn.Sequential(*layers)

    def _layer(self, block, planes, blocks, stride):
        downsample = get_downsampling_layer(self.inplanes, block, planes,
                                            stride)

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        # --------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,)) - batch_mins,
                                 batch_maxs - batch_mins + 1e-10)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed


def get_downsampling_layer(inplanes, block, planes, stride):
    outplanes = planes * block.expansion
    if stride == 1 and inplanes == outplanes:
        return
    else:
        return nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )


def load_pretrained_model(model):
    strict_rule = True

    state_dict = torch.load('resnet50-19c8e357.pth')

    state_dict = remove_layer(state_dict, 'fc')
    strict_rule = False

    model.load_state_dict(state_dict, strict=strict_rule)
    return model


def model(config, pretrained=True):
    model = IDC(Bottleneck, [3, 4, 6, 3], config)
    if pretrained:
        model = load_pretrained_model(model)
    return model
