# using ResNeSt-50 as an example
import torch
from resnest.resnest import resnest50,resnest50_REFINEDET
net = resnest50(pretrained=False)
x = torch.rand(1, 3, 320, 320)
x2 =torch.split(x,2,dim=3)

net.eval()
net =net.cuda()
print(net)
x=x.cuda()
y = net(x)
