import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch

   
class unet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        bilinear = True
        
        self.conv1a   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bnd1a    = nn.BatchNorm2d(64) # channel
        self.conv1b   = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bnd1b    = nn.BatchNorm2d(64) # channel
        self.mp1      = nn.MaxPool2d(2)
        
        self.conv2a   = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bnd2a    = nn.BatchNorm2d(128) # channel
        self.conv2b   = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bnd2b    = nn.BatchNorm2d(128) # channel
        self.mp2      = nn.MaxPool2d(2)
        
        self.conv3a   = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bnd3a    = nn.BatchNorm2d(256) # channel
        self.conv3b   = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bnd3b    = nn.BatchNorm2d(256) # channel
        self.mp3      = nn.MaxPool2d(2)
        
        self.conv4a   = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bnd4a    = nn.BatchNorm2d(512) # channel
        self.conv4b   = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bnd4b    = nn.BatchNorm2d(512) # channel
        self.mp4      = nn.MaxPool2d(2)
        
        self.conv5a   = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bnd5a    = nn.BatchNorm2d(1024) # channel
        self.conv5b   = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bnd5b    = nn.BatchNorm2d(1024) # channel
        
        self.relu    = nn.ReLU(inplace=True)
        ######################################################################################################
        factor = 2 if bilinear else 1
        
        self.up4        = nn.ConvTranspose2d(1024,512, kernel_size=3, stride=2, padding=0, dilation=1, output_padding=1)
        self.upbn4      = nn.BatchNorm2d(512)
        self.deconv4a   = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bn4a       = nn.BatchNorm2d(1024) # channel
        self.deconv4b   = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bn4b       = nn.BatchNorm2d(512) # channel
        
        self.up3        = nn.ConvTranspose2d(512,256, kernel_size=3, stride=2, padding=0, dilation=1, output_padding=1)
        self.upbn3      = nn.BatchNorm2d(256)
        self.deconv3a   = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bn3a       = nn.BatchNorm2d(512) # channel
        self.deconv3b   = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bn3b       = nn.BatchNorm2d(256) # channel
        
        self.up2        = nn.ConvTranspose2d(256,128, kernel_size=3, stride=2, padding=0, dilation=1, output_padding=1)
        self.upbn2      = nn.BatchNorm2d(128)
        self.deconv2a   = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bn2a       = nn.BatchNorm2d(256) # channel
        self.deconv2b   = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bn2b       = nn.BatchNorm2d(128) # channel
        
        self.up1        = nn.ConvTranspose2d(128,64, kernel_size=3, stride=2, padding=0, dilation=1, output_padding=1)
        self.upbn1      = nn.BatchNorm2d(64)
        self.deconv1a   = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bn1a       = nn.BatchNorm2d(128) # channel
        self.deconv1b   = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0, dilation=1) # output batch_size*32*192*384
        self.bn1b       = nn.BatchNorm2d(64) # channel
        
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)

    def forward(self, x):
        # Complete the forward function for the rest of the encoder
        
        x = F.pad(x, [4, 4,
                      4, 4])
        x1a = self.bnd1a(self.relu(self.conv1a(x)))
        x1b = self.bnd1b(self.relu(self.conv1b(x1a)))
        x1m = self.mp1(x1b)
        
        x2a = self.bnd2a(self.relu(self.conv2a(x1m)))
        x2b = self.bnd2b(self.relu(self.conv2b(x2a)))
        x2m = self.mp1(x2b)
        
        x3a = self.bnd3a(self.relu(self.conv3a(x2m)))
        x3b = self.bnd3b(self.relu(self.conv3b(x3a)))
        x3m = self.mp3(x3b)
        
        x4a = self.bnd4a(self.relu(self.conv4a(x3m)))
        x4b = self.bnd4b(self.relu(self.conv4b(x4a)))
        x4m = self.mp4(x4b)
        
        x5a = self.bnd5a(self.relu(self.conv5a(x4m)))
        x5b = self.bnd5b(self.relu(self.conv5b(x5a)))
        
        #############################################################
        
        y4u = self.upbn4(self.relu(self.up4(x5b)))
        y1 = y4u
        y2 = x4b
        diffY = y2.size()[2] - y1.size()[2]
        diffX = y2.size()[3] - y1.size()[3]
        y1 = F.pad(y1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        y = torch.cat([y2, y1], dim=1)
        y4a = self.bn4a(self.relu(self.deconv4a(y)))
        y4b = self.bn4b(self.relu(self.deconv4b(y4a)))
        
        y3u = self.upbn3(self.relu(self.up3(y4b)))
        y1 = y3u
        y2 = x3b
        diffY = y2.size()[2] - y1.size()[2]
        diffX = y2.size()[3] - y1.size()[3]
        y1 = F.pad(y1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        y = torch.cat([y2, y1], dim=1)
        y3a = self.bn3a(self.relu(self.deconv3a(y)))
        y3b = self.bn3b(self.relu(self.deconv3b(y3a)))
        
        y2u = self.upbn2(self.relu(self.up2(y3b)))
        y1 = y2u
        y2 = x2b
        diffY = y2.size()[2] - y1.size()[2]
        diffX = y2.size()[3] - y1.size()[3]
        y1 = F.pad(y1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        y = torch.cat([y2, y1], dim=1)
        y2a = self.bn2a(self.relu(self.deconv2a(y)))
        y2b = self.bn2b(self.relu(self.deconv2b(y2a)))
        
        y1u = self.upbn1(self.relu(self.up1(y2b)))
        y1 = y1u
        y2 = x1b
        diffY = y2.size()[2] - y1.size()[2]
        diffX = y2.size()[3] - y1.size()[3]
        y1 = F.pad(y1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        y = torch.cat([y2, y1], dim=1)
        y1a = self.bn1a(self.relu(self.deconv1a(y)))
        y1b = self.bn1b(self.relu(self.deconv1b(y1a)))
        #print(y1b.size())
        score = self.classifier(y1b)
        #print(score.size())
        
        return score  # size=(N, n_class, x.H/1, x.W/1)
    