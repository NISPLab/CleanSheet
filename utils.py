import torch
import torch.nn as nn
from cifarModels.resnet import cifar10_resnet20
from cifarModels.resnet import cifar10_resnet44
from cifarModels.resnet import cifar10_resnet56
from cifarModels.vgg import cifar10_vgg11_bn
from cifarModels.vgg import cifar10_vgg13_bn
from cifarModels.vgg import cifar10_vgg16_bn
from cifarModels.vgg import cifar10_vgg19_bn
from cifarModels.mobilenetv2 import cifar10_mobilenetv2_x0_5
from cifarModels.mobilenetv2 import cifar10_mobilenetv2_x0_75
from cifarModels.mobilenetv2 import cifar10_mobilenetv2_x1_0
from cifarModels.mobilenetv2 import cifar10_mobilenetv2_x1_4
from cifarModels.shufflenetv2 import cifar10_shufflenetv2_x0_5
from cifarModels.shufflenetv2 import cifar10_shufflenetv2_x1_0
from cifarModels.shufflenetv2 import cifar10_shufflenetv2_x1_5
from cifarModels.shufflenetv2 import cifar10_shufflenetv2_x2_0
from cifarModels.repvgg import cifar10_repvgg_a0
from cifarModels.repvgg import cifar10_repvgg_a1
from cifarModels.repvgg import cifar10_repvgg_a2


modelDict = {
    'mobile0.5':cifar10_mobilenetv2_x0_5,
    'mobile0.75':cifar10_mobilenetv2_x0_75,
    'mobile1.0':cifar10_mobilenetv2_x1_0,
    'mobile1.4':cifar10_mobilenetv2_x1_4,
    'repvgga0':cifar10_repvgg_a0,
    'repvgga1':cifar10_repvgg_a1,
    'repvgga2':cifar10_repvgg_a2,
    'resnet20':cifar10_resnet20,
    'resnet44':cifar10_resnet44,
    'resnet56':cifar10_resnet56,
    'shuffle0.5':cifar10_shufflenetv2_x0_5,
    'shuffle1.0':cifar10_shufflenetv2_x1_0,
    'shuffle1.5':cifar10_shufflenetv2_x1_5,
    'shuffle2.0':cifar10_shufflenetv2_x2_0,
    'vgg11':cifar10_vgg11_bn,
    'vgg13':cifar10_vgg13_bn,
    'vgg16':cifar10_vgg16_bn,
    'vgg19':cifar10_vgg19_bn,  
}

pathDict = {
    'mobile0.5':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_mobilenetv2_x0_5-ca14ced9.pt',
    'mobile0.75':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_mobilenetv2_x0_75-a53c314e.pt',
    'mobile1.0':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_mobilenetv2_x1_0-fe6a5b48.pt',
    'mobile1.4':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_mobilenetv2_x1_4-3bbbd6e2.pt',
    'repvgga0':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_repvgg_a0-ef08a50e.pt',
    'repvgga1':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_repvgg_a1-38d2431b.pt',
    'repvgga2':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_repvgg_a2-09488915.pt',
    'resnet20':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_resnet20-4118986f.pt',
    'resnet44':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_resnet44-2a3cabcb.pt',
    'resnet56':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_resnet56-187c023a.pt',
    'shuffle0.5':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_shufflenetv2_x0_5-1308b4e9.pt',
    'shuffle1.0':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_shufflenetv2_x1_0-98807be3.pt',
    'shuffle1.5':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_shufflenetv2_x1_5-296694dd.pt',
    'shuffle2.0':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_shufflenetv2_x2_0-ec31611c.pt',
    'vgg11':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_vgg11_bn-eaeebf42.pt',
    'vgg13':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_vgg13_bn-c01e4a43.pt',
    'vgg16':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_vgg16_bn-6ee7ea24.pt',
    'vgg19':'/home/ubuntu/Data/Projects/gyj/pretrained/cifar10_vgg19_bn-57191229.pt',    
}

class Trigger(nn.Module):

    def __init__(self, size: int = 32, transparency: float = 1.) -> None:
        super().__init__()

        self.size = size
        self.mask = nn.Parameter(torch.rand(size, size,device=torch.device('cuda')),requires_grad=True)
        self.transparency = transparency
        self.trigger = nn.Parameter(torch.rand(3, size, size,device=torch.device('cuda')) * 4 - 2,requires_grad=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transparency * self.mask * self.trigger + (1 - self.mask * self.transparency) * x
    
class UAP(nn.Module):

    def __init__(self, size: int = 32) -> None:
        super().__init__()

        self.size = size
        self.perturbation = nn.Parameter(torch.zeros(3, size, size,device=torch.device('cuda')),requires_grad=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.perturbation
