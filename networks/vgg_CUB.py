import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import cv2
from skimage import measure
from utils.func import *
from utils.loss import *
def map_process(pros,map,T):
    #T = 3
    weight = F.softmax(pros / T, dim=-1)
    fmap = map * weight.unsqueeze(-1).unsqueeze(-1)
    fmap = torch.sum(fmap, dim=1, keepdim=True)
    return fmap

#VGG-16 Backbone
class IDC(nn.Module):
    def __init__(self, config):
        super(IDC, self).__init__()
        self.config = config
        self.num_classes = config["num_classes"]
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier_cls = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 200, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.classifier_loc = nn.Sequential(
            nn.Conv2d(512+256, 200, kernel_size=3, padding=1),  ## num_classes
        )

    def forward(self, x, label=None):
        conv_copy_4_1 = copy.deepcopy(self.conv4_1)
        relu_copy_4_1 = copy.deepcopy(self.relu4_1)
        conv_copy_4_2 = copy.deepcopy(self.conv4_2)
        relu_copy_4_2 = copy.deepcopy(self.relu4_2)
        conv_copy_4_3 = copy.deepcopy(self.conv4_3)
        relu_copy_4_3 = copy.deepcopy(self.relu4_3)
        conv_copy_5_1 = copy.deepcopy(self.conv5_1)
        relu_copy_5_1 = copy.deepcopy(self.relu5_1)
        conv_copy_5_2 = copy.deepcopy(self.conv5_2)
        relu_copy_5_2 = copy.deepcopy(self.relu5_2)
        conv_copy_5_3 = copy.deepcopy(self.conv5_3)
        relu_copy_5_3 = copy.deepcopy(self.relu5_3)
        classifier_cls_copy = copy.deepcopy(self.classifier_cls)

        self.batch = x.size(0)
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)

        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)

        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x_3 = x.clone()

        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)

        x_4 = x.clone()
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)

        x = self.classifier_cls(x)
        x = self.avg_pool(x).view(x.size(0), -1)
        self.prob1 = x

        p_label = label.unsqueeze(-1)
        self.fmap,self.center = self.tfm_modeling(x_3,x_4,p_label)
        self.bmap = 1 - self.fmap
        x_erase1 = x_3.detach() * (self.fmap)
        x_erase2 = x_3.detach() * (self.bmap)
        x_erase = torch.cat((x_erase1,x_erase2),dim=0)
        x_erase = self.pool3(x_erase)
        x_erase = conv_copy_4_1(x_erase)
        x_erase = relu_copy_4_1(x_erase)
        x_erase = conv_copy_4_2(x_erase)
        x_erase = relu_copy_4_2(x_erase)
        x_erase = conv_copy_4_3(x_erase)
        x_erase = relu_copy_4_3(x_erase)
        x_erase = self.pool4(x_erase)
        x_erase = conv_copy_5_1(x_erase)
        x_erase = relu_copy_5_1(x_erase)
        x_erase = conv_copy_5_2(x_erase)
        x_erase = relu_copy_5_2(x_erase)
        x_erase = conv_copy_5_3(x_erase)
        x_erase = relu_copy_5_3(x_erase)
        x_erase = classifier_cls_copy(x_erase)
        x_erase = self.avg_pool(x_erase).view(x_erase.size(0), -1)
        self.prob2, self.prob3 = torch.chunk(x_erase, 2, dim=0)

        tfm_loss = self.TFM_loss()

        return self.prob1, self.prob2, self.fmap, tfm_loss, self.center

    def evl_forward(self, x):
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)

        x = self.pool1(x)
        x = self.conv2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)

        x = self.pool2(x)
        x = self.conv3_1(x)
        x = self.relu3_1(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x_3 = x.clone()

        x = self.pool3(x)
        x = self.conv4_1(x)
        x = self.relu4_1(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_3(x)

        x_4 = x.clone()
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = self.relu5_1(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)

        x = self.classifier_cls(x)


        x = self.avg_pool(x).view(x.size(0), -1)
        self.prob1 = x
        x_4 = F.interpolate(x_4, x_3.shape[2:], mode='bilinear')
        fmaps_all = self.classifier_loc(torch.cat((x_3, x_4), dim=1))
        self.fmap = map_process(self.prob1, fmaps_all,self.config['tua'])
        self.fmap = F.sigmoid(self.fmap)

        return self.prob1

    def get_expected_correspondence_loc(self,atm):
        B, d, h, w = atm.size()
        grid_n = self.gen_grid(-1, 1, -1, 1, h, w)
        flaten=atm.view(B,d,-1)
        prob = F.softmax(flaten,dim=-1)
        expected_coord_n = torch.sum(grid_n * prob.unsqueeze(-1), dim=2)  # Bxnx2
        return expected_coord_n

    def tfm_modeling(self,x_3,x_4,p_label):
        x_4 = F.interpolate(x_4, x_3.shape[2:], mode='bilinear')
        fmap_all = self.classifier_loc(torch.cat((x_3, x_4), dim=1))
        fmap = torch.zeros(self.batch, 1, x_3.size(-2), x_3.size(-1)).cuda()
        for i in range(self.batch):
            fmap[i][0] = fmap_all[i][p_label[i]].mean(0)
        center = self.get_expected_correspondence_loc(fmap)
        return F.sigmoid(fmap),center

    def TFM_loss(self):
        dis_neg = cos_distance(self.prob1,self.prob3)
        neg_pair = torch.diagonal(dis_neg, dim1=0, dim2=1)
        dis_pos = cos_distance(self.prob1, self.prob2)
        pos_pair = torch.diagonal(dis_pos, dim1=0, dim2=1)
        triple_loss = torch.relu(1 + pos_pair - neg_pair)
        res=triple_loss
        forepixel = self.fmap
        forepixel = forepixel.clone().view(self.batch, -1)
        forepixel = forepixel.mean(1)
        loss = res + forepixel
        loss = loss.mean(0)

        return loss

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        m.weight.data.fill_(0)

def model(config, pretrained=True):
    model = IDC(config)
    if pretrained:
        model.apply(weight_init)
        pretrained_dict = torch.load('vgg16-397923af.pth')
        model_dict = model.state_dict()
        model_conv_name = []

        for i, (k, v) in enumerate(model_dict.items()):
            model_conv_name.append(k)
        for i, (k, v) in enumerate(pretrained_dict.items()):
            if k.split('.')[0] != 'features':
                break
            if np.shape(model_dict[model_conv_name[i]]) == np.shape(v):
                model_dict[model_conv_name[i]] = v
        model.load_state_dict(model_dict)
        print("pretrained weight load complete..")
    return model
