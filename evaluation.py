import os
import sys
import json
import numpy as np
import torch
import yaml
import torchvision.transforms as transforms
from torch.backends import cudnn
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw, ImageFont
import cv2
from utils.func import *
from networks import get_model
from utils.vis import *
from utils.IoU import *
from utils.augment import *
import argparse
from networks import *
from skimage import measure

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='IDC Evaluation on CUB-200-2011 and ILSVRC datasets')
        self.parser.add_argument('--gpu',help='which gpu to use',default='0',dest='gpu')
        self.parser.add_argument('--epoch', type=int, default=39)
        self.parser.add_argument('--configs', type=str, required=True,default='configs/vgg_CUB.yaml')
    def parse(self):
        opt = self.parser.parse_args()
        return opt

args = opts().parse()
with open(args.configs, 'r') as f:
    config = yaml.safe_load(f)
print(config)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
def normalize_map(atten_map,w,h):
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val)/(max_val - min_val)
    atten_norm = cv2.resize(atten_norm, dsize=(w,h))
    return atten_norm
def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data
cudnn.benchmark = True
TEN_CROP = True
normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
transform = transforms.Compose([
        transforms.Resize((config["input_size"],config["input_size"])),
        transforms.CenterCrop(config["crop_size"]),
        transforms.ToTensor(),
        normalize
])
cls_transform = transforms.Compose([
        transforms.Resize((config["input_size"],config["input_size"])),
        transforms.CenterCrop(config["crop_size"]),
        transforms.ToTensor(),
        normalize
])
ten_crop_aug = transforms.Compose([
    transforms.Resize((config["input_size"], config["input_size"])),
    transforms.TenCrop(config["crop_size"]),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
])
model = get_model(config['backbone'])(config,pretrained=False)
model.load_state_dict(torch.load('weights/' + config["backbone"] +'/'+str(args.epoch)+'.pth.tar'))
model = model.to(0)
model.eval()
root = config["dataset_path"]
val_imagedir = os.path.join(root, 'test')

anno_root = os.path.join(root,'bbox')
val_annodir = os.path.join(root, 'test_gt.txt')
val_list_path = os.path.join(root, 'test_list.txt')

classes = os.listdir(val_imagedir)
classes.sort()
temp_softmax = nn.Softmax()

class_to_idx = {classes[i]:i for i in range(len(classes))}

accs = []
accs_top5 = []
loc_accs = []
loc_maxboxaccv2 = []
final_loc = []
final_clsloc = []
final_clsloctop5 = []
bbox_f = open(val_annodir, 'r')
bbox_list = []
for line in bbox_f:
    x0, y0, x1, y1, h, w = line.strip("\n").split(' ')
    x0, y0, x1, y1, h, w = float(x0), float(y0), float(x1), float(y1), float(h), float(w)
    x0, y0, x1, y1 = x0, y0, x1, y1
    bbox_list.append((x0, y0, x1, y1))  ## gt
cur_num = 0
bbox_f.close()

files = [[] for i in range(config["num_classes"])]

with open(val_list_path, 'r') as f:
    for line in f:
        test_img_path, img_class =  line.strip("\n").split(';')
        files[int(img_class)].append(test_img_path)

for k in range(config["num_classes"]):
    cls = classes[k]

    IoUSet = []
    IoUSetTop5 = []
    LocSet = []

    for (i, name) in enumerate(files[k]):

        gt_boxes = bbox_list[cur_num]
        cur_num += 1
        if len(gt_boxes)==0:
            continue

        raw_img = Image.open(os.path.join(val_imagedir, name)).convert('RGB')
        w, h = config["crop_size"],  config["crop_size"]

        with torch.no_grad():
            img = transform(raw_img)
            img = torch.unsqueeze(img, 0)
            img = img.to(0)
            reg_output= model.evl_forward(img)
            
            cam = model.fmap[0][0].data.cpu()
            cam = normalize_map(np.array(cam),w,h)

            highlight = np.zeros(cam.shape)
            highlight[cam > float(config["threshold"])] = 1
            # max component
            all_labels = measure.label(highlight)
            highlight = np.zeros(highlight.shape)
            highlight[all_labels == count_max(all_labels.tolist())] = 1
            highlight = np.round(highlight * 255)
            highlight_big = cv2.resize(highlight, (w, h), interpolation=cv2.INTER_NEAREST)
            CAMs = copy.deepcopy(highlight_big)
            props = measure.regionprops(highlight_big.astype(int))

            if len(props) == 0:
                bbox = [0, 0, w, h]
            else:
                temp = props[0]['bbox']
                bbox = [temp[1], temp[0], temp[3], temp[2]]

            if TEN_CROP:

                img = ten_crop_aug(raw_img)
                img = img.to(0)
                vgg16_out = model.evl_forward(img)
                vgg16_out = nn.Softmax()(vgg16_out)
                vgg16_out = torch.mean(vgg16_out,dim=0,keepdim=True)
                vgg16_out = torch.topk(vgg16_out, 5, 1)[1]
            else:
                img = cls_transform(raw_img)
                img = torch.unsqueeze(img, 0)
                img = img.to(0)
                vgg16_out = model.evl_forward(img)
                vgg16_out = torch.topk(vgg16_out, 5, 1)[1]
            vgg16_out = to_data(vgg16_out)
            vgg16_out = torch.squeeze(vgg16_out)
            vgg16_out = vgg16_out.numpy()
            out = vgg16_out

        #handle resize and centercrop for gt_boxes

        gt_bbox_i = list(gt_boxes)
        raw_img_i = raw_img
        raw_img_i, gt_bbox_i = ResizedBBoxCrop((256,256))(raw_img, gt_bbox_i)
        raw_img_i, gt_bbox_i = CenterBBoxCrop((224))(raw_img_i, gt_bbox_i)
        # w, h = raw_img_i.size
        gt_bbox_i[0] = gt_bbox_i[0] * w
        gt_bbox_i[2] = gt_bbox_i[2] * w
        gt_bbox_i[1] = gt_bbox_i[1] * h
        gt_bbox_i[3] = gt_bbox_i[3] * h

        gt_boxes = gt_bbox_i

        bbox[0] = bbox[0]  
        bbox[2] = bbox[2] 
        bbox[1] = bbox[1] 
        bbox[3] = bbox[3]

        max_iou = -1
        iou = IoU(bbox, gt_boxes)
        if iou > max_iou:
            max_iou = iou

        LocSet.append(max_iou)
        temp_loc_iou = max_iou
        if out[0] != class_to_idx[cls]:
            max_iou = 0

        IoUSet.append(max_iou)
        #cal top5 IoU
        max_iou = 0
        for i in range(5):
            if out[i] == class_to_idx[cls]:
                max_iou = temp_loc_iou
        IoUSetTop5.append(max_iou)
        
    cls_loc_acc = np.sum(np.array(IoUSet) > 0.5) / len(IoUSet)
    final_clsloc.extend(IoUSet)
    cls_loc_acc_top5 = np.sum(np.array(IoUSetTop5) > 0.5) / len(IoUSetTop5)
    final_clsloctop5.extend(IoUSetTop5)
    loc_acc = np.sum(np.array(LocSet) > 0.5) / len(LocSet)
    maxboxaccv2 = (np.sum(np.array(LocSet) > 0.3) / len(LocSet) + np.sum(np.array(LocSet) > 0.5) / len(LocSet) + np.sum(
        np.array(LocSet) > 0.7) / len(LocSet)) / 3
    final_loc.extend(LocSet)
    print('{} cls-loc acc is {}, loc acc is {}, loc acc 5 is {}'.format(cls, cls_loc_acc, loc_acc, cls_loc_acc_top5))
    accs.append(cls_loc_acc)
    accs_top5.append(cls_loc_acc_top5)
    loc_accs.append(loc_acc)
    loc_maxboxaccv2.append(maxboxaccv2)

#f = open("k.txt", "w")
#save_list = list(map(str,final_loc))
#str = '\n'
#f.writelines(str.join(save_list))
#f.close()

print('Cls-Loc acc {}'.format(np.mean(accs)))
print('Cls-Loc acc Top 5 {}'.format(np.mean(accs_top5)))
print('GT Loc acc {}'.format(np.mean(loc_accs)))
print('MaxBoxAccV2 acc {}'.format(np.mean(loc_maxboxaccv2)))