from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from data import voc_refinedet, coco_refinedet
import os

from .resnet50_v1d_86 import *
from resnest import resnest

export_tensorrt_onnx = True
export_trace_pt = True


class RefineDet(nn.Module):

    def  __init__(self, size, base, extras, ARM, ODM, TCB, num_classes):
        super(RefineDet, self).__init__()
        # self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco_refinedet, voc_refinedet)[num_classes==2]  #
        self.priorbox = PriorBox(self.cfg[str(size)])
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.size = size
        #resnet50-86
       # self.conv4_3_L2Norm = L2Norm(280, 10)
       # self.conv5_3_L2Norm = L2Norm(856, 8)
#resnest50
        self.conv4_3_L2Norm = L2Norm(512, 10)
        self.conv5_3_L2Norm = L2Norm(1024, 8)

        self.extras = nn.ModuleList(extras)

        self.arm_loc = nn.ModuleList(ARM[0])
        self.arm_conf = nn.ModuleList(ARM[1])
        self.odm_loc = nn.ModuleList(ODM[0])
        self.odm_conf = nn.ModuleList(ODM[1])
        #self.tcb = nn.ModuleList(TCB)
        self.tcb0 = nn.ModuleList(TCB[0])
        self.tcb1 = nn.ModuleList(TCB[1])
        self.tcb2 = nn.ModuleList(TCB[2])


        # if phase == 'test':
        self.softmax = nn.Softmax(dim=-1)
        self.detect = Detect_RefineDet(num_classes, self.size,bkg_label =  0, top_k = 1000, conf_thresh = 0.01, nms_thresh = 0.45, objectness_thre = 0.01, keep_top_k = 500)
        self.Resnet86 = nn.ModuleList(base)
        #self.model = get_res86_net()
        self.model = get_resnest50("/home/py/Disk700G/2019code/Refinedet_Pytorch-res86/weights/resnest50-528c19ca.pth")



    def forward(self, x):

        sources = list()
        tcb_source = list()
        arm_loc = list()
        arm_conf = list()
        odm_loc = list()
        odm_conf = list()

        for i, k in enumerate(self.model.forward(x)):
            if 0 == i:
                s = self.conv4_3_L2Norm(k)
                sources.append(s)
            elif 1 == i:
                s = self.conv5_3_L2Norm(k)
                sources.append(s)
            else:
                sources.append(k)
        x = sources[2]

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        for (x, l, c) in zip(sources, self.arm_loc, self.arm_conf):
            arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)

        p = None
        for k, v in enumerate(sources[::-1]):
            s = v
            for i in range(3):
                s = self.tcb0[(3-k)*3 + i](s)
                #print(s.size())
            if k != 0:
                u = p
                u = self.tcb1[3-k](u)
                s += u
            for i in range(3):
                s = self.tcb2[(3-k)*3 + i](s)
            p = s
            tcb_source.append(s)
        tcb_source.reverse()

        # apply ODM to source layers
        for (x, l, c) in zip(tcb_source, self.odm_loc, self.odm_conf):
            odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)

        # if self.phase == "test":
            #print(loc, conf)
        # output = self.detect(
        #     arm_loc.view(arm_loc.size(0), -1, 4),  # arm loc preds
        #     self.softmax(arm_conf.view(arm_conf.size(0), -1, 2)),  # arm conf preds
        #     odm_loc.view(odm_loc.size(0), -1, 4),  # odm loc preds
        #     self.softmax(odm_conf.view(odm_conf.size(0), -1, self.num_classes)),  # odm conf preds
        #     self.priors.type(type(x.data))  # default boxes
        # )

        # else:
        output = (
            arm_loc.view(arm_loc.size(0), -1, 4),
            arm_conf.view(arm_conf.size(0), -1, 2),
            odm_loc.view(odm_loc.size(0), -1, 4),
            odm_conf.view(odm_conf.size(0), -1, self.num_classes),
            self.priors
        )
        return output


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')

################################################################################################
base = {
    '320': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras = {
    '320': [256, 'S', 512],
    '512': [256, 'S', 512],
}
mbox = {
    '320': [3, 3, 3, 3],  # number of boxes per feature map location
    '512': [3, 3, 3, 3],  # number of boxes per feature map location
}
tcb = {
    # '320': [88, 224, 1336, 512],
   # '320': [280, 856, 2048, 512],#resnet50-86
    '320': [512, 1024, 2048, 512], #resnest50
    '512': [512, 512, 1024, 512],
}

def resnet86():
    model = get_res86_net()
    layer = []
    for k, v in model._modules.items():
        # print(k)
        layer.append(v)
    return layer



def resnest50():
    model = resnest.resnest50_REFINEDET()
    layer = []
    for k, v in model._modules.items():
        # print(k)
        layer.append(v)
    return layer
def get_resnest50(path):
    model = resnest.resnest50_REFINEDET()
    if path != '':
        model.load_state_dict(torch.load(path), strict=False)
    return model

def add_extras(cfg, size, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],
                           kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def arm_multibox(vgg, extra_layers, cfg):
    arm_loc_layers = []
    arm_conf_layers = []
    #vgg_source = [280, 856, 2048] #resnet50-86
    vgg_source = [512, 1024, 2048] #resnest50

    for k, v in enumerate(vgg_source):
        arm_loc_layers += [nn.Conv2d(vgg_source[k],
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(vgg_source[k],
                        cfg[k] * 2, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 3):
        arm_loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * 2, kernel_size=3, padding=1)]
    return (arm_loc_layers, arm_conf_layers)

def odm_multibox(vgg, extra_layers, cfg, num_classes):
    odm_loc_layers = []
    odm_conf_layers = []
    vgg_source = [60, 99, 165]
    for k, v in enumerate(vgg_source):
        odm_loc_layers += [nn.Conv2d(256, cfg[k] * 4, kernel_size=3, padding=1)]
        odm_conf_layers += [nn.Conv2d(256, cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 3):
        odm_loc_layers += [nn.Conv2d(256, cfg[k] * 4, kernel_size=3, padding=1)]
        odm_conf_layers += [nn.Conv2d(256, cfg[k] * num_classes, kernel_size=3, padding=1)]
    return (odm_loc_layers, odm_conf_layers)

def add_tcb(cfg):
    feature_scale_layers = []
    feature_upsample_layers = []
    feature_pred_layers = []
    for k, v in enumerate(cfg):
        feature_scale_layers += [nn.Conv2d(cfg[k], 256, 3, padding=1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(256, 256, 3, padding=1)
        ]
        feature_pred_layers += [nn.ReLU(inplace=True),
                                nn.Conv2d(256, 256, 3, padding=1),
                                nn.ReLU(inplace=True)
        ]
        if k != len(cfg) - 1:
            feature_upsample_layers += [nn.ConvTranspose2d(256, 256, 2, 2)]
    return (feature_scale_layers, feature_upsample_layers, feature_pred_layers)


def build_refinedet(size, num_classes):
    #base_ = resnet86() #resnet50-86
    base_ =resnest50()
    extras_ = add_extras(extras[str(size)], size, 2048)
    ARM_ = arm_multibox(base_, extras_, mbox[str(size)])
    ODM_ = odm_multibox(base_, extras_, mbox[str(size)], num_classes)
    TCB_ = add_tcb(tcb[str(size)])
    return RefineDet(size, base_, extras_, ARM_, ODM_, TCB_, num_classes)