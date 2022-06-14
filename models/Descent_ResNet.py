import torch.nn as nn
import torch
from torchvision import models


class Descent_ResNet(nn.Module):
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
        # downsample make size small
        self.deconv2 = nn.ConvTranspose2d(256,128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(128)
        
        self.deconv3 = nn.ConvTranspose2d(128,64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(64)
        
        self.deconv4 = nn.ConvTranspose2d(64,32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(32)
        
        self.deconv5 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(16)
        
        self.classifier = nn.Conv2d(16, self.n_class, kernel_size=1)
        
        
        # skip connection
        self.downsample_y1 = nn.Conv2d(512, 256, kernel_size=1)
        self.downsample_x1 = nn.Conv2d(64, 32, kernel_size=1)
        
        
    def forward(self, images):
        
        with torch.no_grad(): # we don't want to ruin that pretained weights
            # I cannot remove avgpool with success :( even I remove avgpool it still return weirdly shaped outputs
            x1 = self.resnet.conv1(images) # [16, 64, 192, 384]
            x2 = self.resnet.maxpool(self.resnet.relu(self.resnet.bn1(x1))) # max pool change size # [16, 64, 96, 192]
            x3 = self.resnet.layer1(x2) # [16, 64, 96, 192]
            x4 = self.resnet.layer2(x3) # [16, 128, 48, 96]
            x5 = self.resnet.layer3(x4) # [16, 256, 24, 48]
            features = self.resnet.layer4(x5) # [4, 512, 12, 24] looks more right to me this stupid thing # [16, 512, 12, 24]
            
            
        
    
        y1 = self.bn1(self.relu(self.deconv1(features)))    # [16, 512, 24, 48] #x5
        # add skip connections
        y1_p = self.downsample_y1(y1)+x5 # 256 channel
        
        
        
        # Complete the forward function for the rest of the decoder 
        y2 = self.bn2(self.relu(self.deconv2(y1_p)))   # [16, 128, 48, 96] #x4
        y2_p = y2+x4 # [16, 128, 48, 96]
        
        
        y3 = self.bn3(self.relu(self.deconv3(y2_p)))   # [16, 64, 96, 192] #x3
        y3_p = y3+x3
        
        y4 = self.bn4(self.relu(self.deconv4(y3_p)))   # [16, 32, 192, 384] #x3
        
        y4_p = y4+self.downsample_x1(x1)
        
        
        
        
        out_decoder = self.bn5(self.relu(self.deconv5(y4_p)))  
        
    
    
        score = self.classifier(out_decoder)     
        
        
        return score # [batch size, 10, 384, 768]
