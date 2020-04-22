# using ResNeSt-50 as an example
import torch
from resnest.torch import resnest50,resnest50_REFINEDET
net = resnest50_REFINEDET(pretrained=False)
x = torch.rand(1, 3, 320, 320)
net.eval()
print(net)
y = net(x)
