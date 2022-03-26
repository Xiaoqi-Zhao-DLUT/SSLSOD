import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

class Crossmodal_Autoendoer(nn.Module):

    def __init__(self):
        super(Crossmodal_Autoendoer, self).__init__()
        ################################vgg16#######################################
        ##set 'pretrained=False' for SSL model or 'pretrained=True' for ImageNet pretrained model.
        feats = list(models.vgg16_bn(pretrained=False).features.children())
        feats1 = list(models.vgg16_bn(pretrained=False).features.children())
        #self.conv0 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv1_RGB = nn.Sequential(*feats[0:6])
        self.conv2_RGB = nn.Sequential(*feats[6:13])
        self.conv3_RGB = nn.Sequential(*feats[13:23])
        self.conv4_RGB = nn.Sequential(*feats[23:33])
        self.conv5_RGB = nn.Sequential(*feats[33:43])

        self.conv1_depth = nn.Sequential(*feats1[0:6])
        self.conv2_depth = nn.Sequential(*feats1[6:13])
        self.conv3_depth = nn.Sequential(*feats1[13:23])
        self.conv4_depth = nn.Sequential(*feats1[23:33])
        self.conv5_depth = nn.Sequential(*feats1[33:43])

        self.output4_rgb = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.output3_rgb = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.output2_rgb = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.PReLU())
        self.output1_rgbtod = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

        self.output4_depth = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.output3_depth = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.output2_depth = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.PReLU())
        self.output1_depthtorgb = nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, padding=1))

        self.sideout5_rgbtod =  nn.Sequential(nn.Conv2d(512, 1, kernel_size=3, padding=1))
        self.sideout4_rgbtod =  nn.Sequential(nn.Conv2d(256, 1, kernel_size=3, padding=1))
        self.sideout3_rgbtod =  nn.Sequential(nn.Conv2d(128, 1, kernel_size=3, padding=1))
        self.sideout2_rgbtod =  nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))

        self.sideout5_depthtorgb =  nn.Sequential(nn.Conv2d(512, 3, kernel_size=3, padding=1))
        self.sideout4_depthtorgb =  nn.Sequential(nn.Conv2d(256, 3, kernel_size=3, padding=1))
        self.sideout3_depthtorgb =  nn.Sequential(nn.Conv2d(128, 3, kernel_size=3, padding=1))
        self.sideout2_depthtorgb =  nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, padding=1))



        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True

    def forward(self, x,depth):

        input = x
        B,_,_,_ = input.size()
        e1_rgb = self.conv1_RGB(x)
        e1_depth = self.conv1_depth(depth)
        e2_rgb = self.conv2_RGB(e1_rgb)
        e2_depth= self.conv2_depth(e1_depth)
        e3_rgb = self.conv3_RGB(e2_rgb)
        e3_depth = self.conv3_depth(e2_depth)
        e4_rgb = self.conv4_RGB(e3_rgb)
        e4_depth = self.conv4_depth(e3_depth)
        e5_rgb = self.conv5_RGB(e4_rgb)
        e5_depth = self.conv5_depth(e4_depth)

        sideout5_rgbtod = self.sideout5_rgbtod(e5_rgb)
        output4_rgb = self.output4_rgb(F.upsample(e5_rgb, size=e4_rgb.size()[2:], mode='bilinear')+e4_rgb)
        sideout4_rgbtod = self.sideout4_rgbtod(output4_rgb)
        output3_rgb = self.output3_rgb(F.upsample(output4_rgb, size=e3_rgb.size()[2:], mode='bilinear') + e3_rgb)
        sideout3_rgbtod = self.sideout3_rgbtod(output3_rgb)
        output2_rgb = self.output2_rgb(F.upsample(output3_rgb, size=e2_rgb.size()[2:], mode='bilinear') + e2_rgb)
        sideout2_rgbtod = self.sideout2_rgbtod(output2_rgb)
        sideout1_rgbtod = self.output1_rgbtod(F.upsample(output2_rgb, size=e1_rgb.size()[2:], mode='bilinear') + e1_rgb)

        sideout5_dtorgb = self.sideout5_depthtorgb(e5_depth)
        output4_d = self.output4_depth(F.upsample(e5_depth, size=e4_rgb.size()[2:], mode='bilinear')+e4_depth)
        sideout4_dtorgb = self.sideout4_depthtorgb(output4_d)
        output3_d = self.output3_depth(F.upsample(output4_d, size=e3_rgb.size()[2:], mode='bilinear') + e3_depth)
        sideout3_dtorgb = self.sideout3_depthtorgb(output3_d)
        output2_d = self.output2_depth(F.upsample(output3_d, size=e2_rgb.size()[2:], mode='bilinear') + e2_depth)
        sideout2_dtorgb = self.sideout2_depthtorgb(output2_d)
        sideout1_dtorgb = self.output1_depthtorgb(F.upsample(output2_d, size=e1_rgb.size()[2:], mode='bilinear') + e1_depth)


        if self.training:
            return sideout5_rgbtod,sideout4_rgbtod,sideout3_rgbtod,sideout2_rgbtod,sideout1_rgbtod,sideout5_dtorgb,sideout4_dtorgb,sideout3_dtorgb,sideout2_dtorgb,sideout1_dtorgb
        # return F.sigmoid(sideout1_rgbtod), F.sigmoid(sideout1_dtorgb)
        return e5_rgb,e4_rgb,e3_rgb,e2_rgb,e1_rgb, e5_depth,e4_depth,e3_depth,e2_depth,e1_depth
        # return e5_rgb,e4_rgb,e3_rgb,e2_rgb,e1_rgb

