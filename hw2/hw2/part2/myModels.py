
# Modelzoo for usage 
# Feel free to add any model you like for your final result
# Note : Pretrained model is allowed iff it pretrained on ImageNet

import torch
import torch.nn as nn
import torchvision

class myLeNet(nn.Module):
    def __init__(self, num_out):
        super(myLeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3,6,kernel_size=5, stride=1),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),
                             )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,kernel_size=5),
                             nn.ReLU(),
                             nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.fc1 = nn.Sequential(nn.Linear(400, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84), nn.ReLU())
        self.fc3 = nn.Linear(84,num_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # It is important to check your shape here so that you know how manys nodes are there in first FC in_features
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out

    
class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(residual_block, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=stride),
                                   nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU()
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        
    def forward(self,x):
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

        
class myResNet18(nn.Module):
    def __init__(self, in_channels=3, num_out=10):
        super(myResNet18, self).__init__()
        
        self.block = residual_block
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, 64, 64, 2)
        self.layer2 = self._make_layer(self.block, 64, 128, 2)
        self.layer3 = self._make_layer(self.block, 128, 256, 2)
        self.layer4 = self._make_layer(self.block, 256, 512, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_out)



        # self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        
        ## TO DO ##
        # Define your own residual network here. 
        # Note: You need to use the residual block you design. It can help you a lot in training.
        # If you have no idea how to design a model, check myLeNet provided by TA above.
        
        pass

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                    stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)

        
    def forward(self,x):
        ## TO DO ## 
        # Define the data path yourself by using the network member you define.
        # Note : It's important to print the shape before you flatten all of your nodes into fc layers.
        # It help you to design your model a lot. 
        # x = x.flatten(x)
        # print(x.shape)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        # out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)

        return out

class myResNet34(nn.Module):
    def __init__(self, in_channels=3, num_out=10):
        super(myResNet34, self).__init__()
        
        self.block = residual_block
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.block, 64, 64, 3)
        self.layer2 = self._make_layer(self.block, 64, 128, 4)
        self.layer3 = self._make_layer(self.block, 128, 256, 6)
        self.layer4 = self._make_layer(self.block, 256, 512, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_out)



        # self.stem_conv = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        
        ## TO DO ##
        # Define your own residual network here. 
        # Note: You need to use the residual block you design. It can help you a lot in training.
        # If you have no idea how to design a model, check myLeNet provided by TA above.
        
        pass

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                    stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        
        return nn.Sequential(*layers)

        
    def forward(self,x):
        ## TO DO ## 
        # Define the data path yourself by using the network member you define.
        # Note : It's important to print the shape before you flatten all of your nodes into fc layers.
        # It help you to design your model a lot. 
        # x = x.flatten(x)
        # print(x.shape)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        # out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)

        return out
        
class ResNet18(nn.Module):
    def __init__(self, num_out=10):
        super(ResNet18, self).__init__()
        
        self.resize = torchvision.transforms.Resize(224)
        self.pretrained_model = torchvision.models.resnet18(pretrained=True)
        self.fc = nn.Linear(1000, num_out)

    def forward(self, x):
        out = self.resize(x)
        out = self.pretrained_model(out)
        out = self.fc(out)
        return out


class ResNet34(nn.Module):
    def __init__(self, num_out=10):
        super(ResNet34, self).__init__()
        
        self.resize = torchvision.transforms.Resize(224)
        self.pretrained_model = torchvision.models.resnet34(pretrained=True)
        self.fc = nn.Linear(1000, num_out)

    def forward(self, x):
        out = self.resize(x)
        out = self.pretrained_model(out)
        out = self.fc(out)
        return out