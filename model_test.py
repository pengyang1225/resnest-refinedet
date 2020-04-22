from __future__ import print_function
import torch

import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOCAnnotationTransform, VOCDetection, BaseTransform
from models.refinedet import build_refinedet
# from models.refinedet_test import build_refinedet

import numpy as np
import cv2
import os

def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == '__main__':

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    num_classes = 2# len(labelmap) + 1                      # +1 for background
    net = build_refinedet(320, num_classes)            # initialize SSD
    net.load_state_dict(torch.load("/home/pengyang/Code/gitlab/follow_model_tx2/pytorch_model_pth/resnet50-86_refinedet/RefineDet320_VOC_res86_body_head.pth"))
    net.eval()
    net = net.cuda()
    cudnn.benchmark = True

    print('Finished loading model!')

    VOC_ROOT = "/home/py/Disk700G/2019code/Train_data/VOCdevkit/"
    src = os.path.join(VOC_ROOT, "VOC2007/ImageSets/Main")
    imgPath = os.path.join(VOC_ROOT, "VOC2007/JPEGImages")

    dst = "/home/lzm/Disk2/work_dl/Pytorch_refinedet/Data_dir/Test_model/RefineDet320_VOC_140000_withhead"
    check_dir(dst)

    for dirs in os.listdir(src):
        dstPath = os.path.join(dst, dirs.rstrip('.txt'))
        check_dir(dstPath)

        txt_path = os.path.join(src, dirs)
        listA = open(txt_path, 'r').readlines()

        dataset = VOCDetection(VOC_ROOT, [('2007', dirs.rstrip('.txt'))],
                               BaseTransform(int(320), (104, 117, 123)),
                               VOCAnnotationTransform())

        for i in range(len(listA)):
            im, gt, h, w = dataset.pull_item(i)
            img = cv2.imread(os.path.join(imgPath, dataset.ids[i][1]+'.jpg'))

            x = Variable(im.unsqueeze(0))
            x = x.cuda()
            detections = net(x).data

            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                item = dets.cpu().numpy()
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32, copy=False)
                count = 0
                for k in item:
                    if float(k[0]) >= 0.9:
                        count += 1
                        xmin = int(k[1] * w)
                        ymin = int(k[2] * h)
                        xmax = int(k[3] * w)
                        ymax = int(k[4] * h)

                        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                        cv2.rectangle(img, (xmin, ymin + 30), (xmin + 120, ymin + 2), (255, 128, 128), -1)
                        cv2.putText(img, '1 : ' + '%.2f' % k[0], (xmin, ymin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, False)
                results = os.path.join(dstPath, str(count))
                check_dir(results)
                cv2.imwrite(results + '/' +dataset.ids[i][1]+'.jpg', img)
                print(dirs.rstrip('.txt'), ":", i, "->", len(listA), ":", dataset.ids[i][1]+'.jpg')
