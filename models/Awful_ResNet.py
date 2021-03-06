import torch.nn as nn
import torch
from torchvision import models
class Awful_ResNet(nn.Module):
    ''' A terrible ResNet that basically copies resnet18 (as encoder) and the basic_fcn decoder'''
    def __init__(self, n_class, resnet_n = 18):
        super().__init__()
        # steal from pretained reset
        
        self.n_class = n_class
        
        if resnet_n == 18:
            self.resnet=models.resnet18(pretrained=True)
        elif resnet_n==34:
            self.resnet=models.resnet34(pretrained=True)
        else:
            self.resnet=models.resnet152(pretrained=True)
        # 
        # remove resnet fully connected (fc) layer
        old_output_size = self.resnet.fc.in_features
        # save the original dimension size
        self.resnet.fc = nn.Identity() # if only remove this become [4, 512]
        self.resnet.avgpool = nn.Identity() #you  are supposed to remove this, but  become [4, 147456]'
        
        
        
        
        # here's the decode my friend
        # no residual connections yet my friend
        # copy pasting from basic_fnc
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(old_output_size, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512,256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256,128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128,64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)
        
    def encoder(self, images):
        
        with torch.no_grad(): # we don't want to ruin that pretained weights
            # I cannot remove avgpool with success :( even I remove avgpool it still return weirdly shaped outputs
            x1 = self.resnet.conv1(images)
            x1.shape # batchsize, 64 channels, size of img
            x2 = self.resnet.maxpool(self.resnet.relu(self.resnet.bn1(x1))) # max pool change size
            x3 = self.resnet.layer1(x2)
            x4 = self.resnet.layer2(x3)
            x5 = self.resnet.layer3(x4)
            features = self.resnet.layer4(x5) # [4, 512, 12, 24] looks more right to me this stupid thing
        return features
    def decoder(self, features):
        ''' does decoder have to look the same as the encoder?? do they??'''
        # copy and paste from basic_fcn
        # here we want grad. Can we also use some decoder to make a frankenstein NN?
        y1 = self.bn1(self.relu(self.deconv1(features)))    
        # Complete the forward function for the rest of the decoder
        y2 = self.bn2(self.relu(self.deconv2(y1)))   
        y3 = self.bn3(self.relu(self.deconv3(y2)))   
        y4 = self.bn4(self.relu(self.deconv4(y3)))   
        out_decoder = self.bn5(self.relu(self.deconv5(y4)))  
        
        return out_decoder
    
    def forward(self, x):
        features = self.encoder(x)
        out_decoder = self.decoder(features)
        # the we classify holy moly
        score = self.classifier(out_decoder)     
        
        
        return score # [batch size, 10, 384, 768]