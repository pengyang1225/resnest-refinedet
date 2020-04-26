# resnest-refinedet
An improved version of refinedet network, modify the backbone network, and provide resnet50 compression network and resnest50 method

A higher performance [PyTorch](http://pytorch.org/) implementation of [Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/abs/1711.06897 ). The official and original Caffe code can be found [here](https://github.com/sfzhang15/RefineDet).

###简介
本实验我没有在VOC上进行测试对比，针对于实际项目实验发现：resnet50-86-refinedet的检测网络，耗时方面较原本vgg时间减少了一半，resnest50的性能较resnet50也有很大的提升。
###
### Table of Contents
- <a href='#performance'>Performance</a>
- <a href='#installation'>Installation</a>
- <a href='#datasets'>Datasets</a>
- <a href='#training-refinedet'>Train</a>
- <a href='#evaluation'>Evaluate</a>
- <a href='#todo'>Future Work</a>
- <a href='#references'>Reference</a>

&nbsp;
&nbsp;
&nbsp;
&nbsp;

## Performance

#### VOC2007 Test

##### mAP (*Single Scale Test*)

| Arch | Paper | Caffe Version | Our PyTorch Version |
|:-:|:-:|:-:|:-:|
| RefineDet320 | 80.0% | 79.52% | 79.81% |
| RefineDet512 | 81.8% | 81.85% | 80.50% |


## Installation
- Install [PyTorch](http://pytorch.org/) by selecting your environment on the website and running the appropriate command.
  * Note: You should use at least PyTorch1.0.0
- Clone this repository.
  * Note: We currently only support Python 3+.
- Then download the dataset by following the [instructions](#datasets) below.
- We now support [Visdom](https://github.com/facebookresearch/visdom) for real-time loss visualization during training!
  * To use Visdom in the browser:
  ```Shell
  # First install Python server and client
  pip install visdom
  # Start the server (probably in a screen or tmux)
  python -m visdom.server
  ```
  * Then (during training) navigate to http://localhost:8097/ (see the Train section below for training details).
- Note: For training, we currently support [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](http://mscoco.org/), and aim to add [ImageNet](http://www.image-net.org/) support soon.

## Datasets
To make things easy, we provide bash scripts to handle the dataset downloads and setup for you.  We also provide simple dataset loaders that inherit `torch.utils.data.Dataset`, making them fully compatible with the `torchvision.datasets` [API](http://pytorch.org/docs/torchvision/datasets.html).


### COCO
Microsoft COCO: Common Objects in Context

##### Download COCO 2014
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/COCO2014.sh
```

### VOC Dataset
PASCAL VOC: Visual Object Classes

##### Download VOC2007 trainval & test
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

##### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```

## Training RefineDet
- First download the fc-reduced [Resnest](https://hangzhang.org/files/resnest.pdf ) PyTorch base network weights at:              https://hangzh.s3.amazonaws.com/encoding/models/resnest50-528c19ca.pth
- and, I have downloaded the file in the `resnest-refinedet/weights` dir:[resnest50](https://pan.baidu.com/s/1Yw3TXtP7SHQbIEB1E3xuPA);秘钥4a7k



- To train resnest50-refinedet . You can manually change them as you want.

```Shell
python train_refinedet.py
```
- Observe the training loss 
```Shell
python -m visdom.server
 Open the web input website: http://localhost:8097/
```
- Note:
  * For training, an NVIDIA GPU is strongly recommended for speed.
  * For instructions on Visdom usage/installation, see the <a href='#installation'>Installation</a> section.
  * You can pick-up training from a checkpoint by specifying the path as one of the training parameters (again, see `train_refinedet.py` for options)

## Evaluation
To evaluate a trained network:

```Shell
python model_test_one_image.py
```

You can specify the parameters listed in the `model_test_one_image.py` file by flagging them or manually changing them.  

## TODO
We have accumulated the following to-do list, which we hope to complete in the near future
- Still to come:
  * [ ] Support for multi-scale testing

## References
- [Original Implementation (CAFFE)](https://github.com/sfzhang15/RefineDet)


 *[ zhanghang1989 /ResNeSt ] (https://github.com/zhanghang1989/ResNeSt)

