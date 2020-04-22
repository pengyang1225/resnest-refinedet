from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from math import sqrt as sqrt
from itertools import product as product
import torch.nn.init as init

import torch.backends.cudnn as cudnn
import os
import torch


export_tensorrt_onnx = True
export_trace_pt = True

import argparse
import numpy as np
import cv2


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=3, help='the number of class')
    parser.add_argument('--number_person', type=int, default=1, help='the number of person')#用于统计单双人
    parser.add_argument('--trained_model', default="/home/pengyang/Desktop/weight/RefineDet320_VOC_final.pth")
    parser.add_argument('--src_path', default="/home/pengyang/Downloads/98_jpg")
    parser.add_argument('--dst_path', default="/home/pengyang/Downloads/98_jpg")

    return parser.parse_args()

args = parse_args()

#################################################################################################
# RefineDet CONFIGS
voc_refinedet = {
    '320': {
        'num_classes': 2,
        'lr_steps': (40000, 60000, 80000),
        'max_iter': 120000,
        'feature_maps': [40, 20, 10, 5],
        'min_dim': 320,
        'steps': [8, 16, 32, 64],
        'min_sizes': [32, 64, 128, 256],
        'max_sizes': [],
        'aspect_ratios': [[2], [2], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'RefineDet_VOC_320',
    },
    '512': {
        'num_classes': 2,
        'lr_steps': (80000, 100000, 120000),
        'max_iter': 120000,
        'feature_maps': [64, 32, 16, 8],
        'min_dim': 512,
        'steps': [8, 16, 32, 64],
        'min_sizes': [32, 64, 128, 256],
        'max_sizes': [],
        'aspect_ratios': [[2], [2], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,
        'name': 'RefineDet_VOC_512',
    }
}

coco_refinedet = {
    'num_classes': 2,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

g_size = 320

variance = voc_refinedet[str(g_size)]['variance']
################################################################################################# Detect_RefineDet
class PriorBox(object):
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                if self.max_sizes:
                    s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out




def batch_decode(loc, priors):
    # jit
    # boxes = torch.cat((
    #     priors[:, :2] + loc[:, :2] * variance[0] * priors[:, 2:],
    #     priors[:, 2:] * torch.exp(loc[:, 2:] * variance[1])), 1)
    boxes = torch.cat((
        priors[:, :, :2] + loc[:, :, :2] * variance[0] * priors[:, :, 2:],
        priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variance[1])), 2)
    # xywh -> xyxy 以下的写法会无法记录计算图
    # boxes[:, :, :2] = boxes[:, :, :2] - boxes[:, :, 2:] * 0.5
    # boxes[:, :, 2:] = boxes[:, :, 2:] + boxes[:, :, :2]

    x2 = boxes[:, :, :2] - boxes[:, :, 2:] * 0.5
    y2 = boxes[:, :, 2:] + x2
    bboxes_in_out = torch.cat((x2, y2), dim=2)
    return bboxes_in_out

def batch_center_size(boxes):
    return torch.cat([(boxes[:, :, 2:] + boxes[:, :, :2])/2,  # cx, cy
                     boxes[:, :, 2:] - boxes[:, :, :2]], 2)  # w, h



class Detect_RefineDet(nn.Module):

    def __init__(self, num_classes, size, bkg_label, top_k, conf_thresh, nms_thresh,
                objectness_thre, keep_top_k):
        super(Detect_RefineDet, self).__init__()
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        self.objectness_thre = objectness_thre
        self.variance = voc_refinedet[str(size)]['variance']

    def forward(self, arm_loc_data, arm_conf_data, odm_loc_data, odm_conf_data, prior_data):
        # [batch, box num, 4]
        loc_data = odm_loc_data
        conf_data = odm_conf_data
        prior_data = prior_data.to(loc_data.device)
        # arm_object_conf = arm_conf_data.data[:, :, 1:]
        # no_object_index = arm_object_conf <= self.objectness_thre
        # conf_data[no_object_index.expand_as(conf_data)] = 0

        num = int(loc_data.size(0))  # batch size
        # prior_data:[box num, 4]
        num_priors = int(prior_data.size(0))
        prior_data = prior_data.unsqueeze(0)
        prior_data = prior_data.repeat(num,1,1)

        # [batch, num_classes,box num]
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)
        # output = torch.zeros(num, self.num_classes, self.top_k, 5)

        default = batch_decode(arm_loc_data, prior_data)

        default = batch_center_size(default)
        decoded_boxes = batch_decode(loc_data, default)

        # decoded_boxes shape  [batch, box num, 4]
        # conf_preds shape   [batch, label num, box num]

        # remove background
        probs = conf_preds[:, 1:, :]
        # print(decoded_boxes)
        # print(probs)
        if export_tensorrt_onnx:
            return decoded_boxes,probs


class RefineDet(nn.Module):

    def __init__(self, size, base, extras, ARM, ODM, TCB, num_classes):
        super(RefineDet, self).__init__()
        # self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco_refinedet, voc_refinedet)[num_classes==3]  #
        self.priorbox = PriorBox(self.cfg[str(size)])
        with torch.no_grad():
            self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.conv4_3_L2Norm = L2Norm(512, 10)
        self.conv5_3_L2Norm = L2Norm(512, 8)
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

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        tcb_source = list()
        arm_loc = list()
        arm_conf = list()
        odm_loc = list()
        odm_conf = list()
        x = x.permute(0, 3, 1, 2)
        mean = torch.tensor((104., 117., 123.), device=x.device)[None, :, None, None]
        x = x - mean
        # apply vgg up to conv4_3 relu and conv5_3 relu
        for k in range(30):
            x = self.vgg[k](x)
            if 22 == k:
                s = self.conv4_3_L2Norm(x)
                sources.append(s)
            elif 29 == k:
                s = self.conv5_3_L2Norm(x)
                sources.append(s)

        # apply vgg up to fc7
        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply ARM and ODM to source layers
        for (x, l, c) in zip(sources, self.arm_loc, self.arm_conf):
            arm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            arm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        arm_loc = torch.cat([o.view(o.size(0), -1) for o in arm_loc], 1)
        arm_conf = torch.cat([o.view(o.size(0), -1) for o in arm_conf], 1)
        #print([x.size() for x in sources])
        # calculate TCB features
        #print([x.size() for x in sources])
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
        #print([x.size() for x in tcb_source])
        tcb_source.reverse()

        # apply ODM to source layers
        for (x, l, c) in zip(tcb_source, self.odm_loc, self.odm_conf):
            odm_loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            odm_conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        odm_loc = torch.cat([o.view(o.size(0), -1) for o in odm_loc], 1)
        odm_conf = torch.cat([o.view(o.size(0), -1) for o in odm_conf], 1)
        #print(arm_loc.size(), arm_conf.size(), odm_loc.size(), odm_conf.size())

        # if self.phase == "test":
            #print(loc, conf)
        output = self.detect(
            arm_loc.view(arm_loc.size(0), -1, 4),  # arm loc preds
            self.softmax(arm_conf.view(arm_conf.size(0), -1, 2)),  # arm conf preds
            odm_loc.view(odm_loc.size(0), -1, 4),  # odm loc preds
            self.softmax(odm_conf.view(odm_conf.size(0), -1, self.num_classes)),  # odm conf preds
            self.priors.type(type(x.data))  # default boxes
        )

        # else:
        #     output = (
        #         arm_loc.view(arm_loc.size(0), -1, 4),
        #         arm_conf.view(arm_conf.size(0), -1, 2),
        #         odm_loc.view(odm_loc.size(0), -1, 4),
        #         odm_conf.view(odm_conf.size(0), -1, self.num_classes),
        #         self.priors
        #     )
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
    '320': [512, 512, 1024, 512],
    '512': [512, 512, 1024, 512],
}

###########################################################################################build_refinedet_test
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]#, ceil_mode=True
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=3, dilation=3)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

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
    vgg_source = [21, 28, -2]
    for k, v in enumerate(vgg_source):
        arm_loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        arm_conf_layers += [nn.Conv2d(vgg[v].out_channels,
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
    vgg_source = [21, 28, -2]
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

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
def is_image_file(filename):
    """Checks if a file is an image.
      Args:
          filename (string): path to a file
      Returns:
          bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def build_refinedet_test(size, num_classes):
    base_ = vgg(base[str(size)], 3)
    extras_ = add_extras(extras[str(size)], size, 1024)
    ARM_ = arm_multibox(base_, extras_, mbox[str(size)])
    ODM_ = odm_multibox(base_, extras_, mbox[str(size)], num_classes)
    TCB_ = add_tcb(tcb[str(size)])
    return RefineDet(size, base_, extras_, ARM_, ODM_, TCB_, num_classes)


################################################################################################
#from torchsummary import summary
if __name__ == '__main__':

    net = build_refinedet_test(g_size, args.num_classes)
    net.load_state_dict(torch.load(args.trained_model))

    net.eval()
    net = net.cuda()
    #summary(net,(3,320,320))
    print(net)
    for name in net.state_dict():
        print(name)


    cudnn.benchmark = True

    print('Finished loading model!')

    src = args.src_path
    dst = args.dst_path



    for id, name in enumerate(os.listdir(src)):
        pic = os.path.join(src, name)
        if not is_image_file(pic):
            continue
        img = cv2.imread('/home/pengyang/Downloads/98_jpg/1_31746E6D0423F603_2019-12-02-10-42-04-060_0_107_7.jpg')
        h, w, c = img.shape

        x = cv2.resize(img, (320, 320)).astype(np.float32)
        # x -= np.array((104, 117, 123), dtype=np.float32)
        # x = x.astype(np.float32)
        # x = x[:, :, (2, 1, 0)]
        im = torch.from_numpy(x)#.permute(2, 0, 1)

        x = im.unsqueeze(0)
        # x = torch.cat((x, x), 0)
        x = x.cuda()
        with torch.no_grad():
            if export_trace_pt:
                traced_model = torch.jit.trace(net, (x,))
                print(traced_model.graph)
                traced_model.save("refinedet.pt")

            import time

            start = time.clock()
            bboxes,scores_out = net(x) #selected_indices,
            end = time.clock()
            print('Running time: %s Seconds' % (end - start))
        if export_tensorrt_onnx:

            torch_out = torch.onnx._export(net, x, "refinedet.onnx", export_params=True)
            break


        # print(id, "->", len(os.listdir(pic)), ":", name)
