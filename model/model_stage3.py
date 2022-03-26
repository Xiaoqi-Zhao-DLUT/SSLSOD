import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

class RGBD_sal(nn.Module):

    def __init__(self):
        super(RGBD_sal, self).__init__()
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

        ################################vgg16#######################################
        self.fuse1_conv1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.fuse1_conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.fuse1_conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.fuse1_conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.fuse1_conv5 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())

        self.fuse1_conv1_fpn = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.fuse1_conv2_fpn = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.fuse1_conv3_fpn = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.fuse1_conv4_fpn = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.fuse1_conv5_fpn = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())

        self.fuse2_conv1 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.fuse2_conv2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.fuse2_conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.fuse2_conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.fuse2_conv5 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())

        self.fuse2_conv1_fpn = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.fuse2_conv2_fpn = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.fuse2_conv3_fpn = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.fuse2_conv4_fpn = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.fuse2_conv5_fpn = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())

        self.fuse3_conv1 =  nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.fuse3_conv2 =  nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.fuse3_conv3 =  nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.fuse3_conv4 =  nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.fuse3_conv5 =  nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())

        self.fuse3_conv1_fpn = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.fuse3_conv2_fpn = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.fuse3_conv3_fpn = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.fuse3_conv4_fpn = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.fuse3_conv5_fpn = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())

        self.fuse4_conv1 =  nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.fuse4_conv2 =  nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.fuse4_conv3 =  nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.fuse4_conv4 =  nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.fuse4_conv5 =  nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())

        self.fuse4_conv1_fpn =  nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.fuse4_conv2_fpn =  nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.fuse4_conv3_fpn =  nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.fuse4_conv4_fpn =  nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.fuse4_conv5_fpn =  nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())

        self.fuse5_conv1 =  nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.fuse5_conv2 =  nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.fuse5_conv3 =  nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.fuse5_conv4 =  nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())
        self.fuse5_conv5 =  nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.PReLU())

        self.sideout5 =  nn.Sequential(nn.Conv2d(512, 1, kernel_size=3, padding=1))
        self.sideout4 =  nn.Sequential(nn.Conv2d(256, 1, kernel_size=3, padding=1))
        self.sideout3 =  nn.Sequential(nn.Conv2d(128, 1, kernel_size=3, padding=1))
        self.sideout2 =  nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))


        self.output4 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.output3 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.output2 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1),nn.BatchNorm2d(64), nn.PReLU())
        self.output1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))


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
        # e5_depth = self.depth_ASPP(e5_depth)

        # print(e5_rgb.shape,e5_depth.shape)
        certain_feature5 = e5_rgb * e5_depth
        fuse5_conv1 = self.fuse5_conv1(e5_rgb+certain_feature5)
        fuse5_conv2 = self.fuse5_conv2(e5_depth+certain_feature5)
        fuse5_certain = self.fuse5_conv3(fuse5_conv1+fuse5_conv2)
        # uncertain_feature5_noconv = torch.abs(e5_rgb - e5_depth)
        uncertain_feature5 = self.fuse5_conv4(torch.abs(e5_rgb - e5_depth))
        fuse5 = self.fuse5_conv5(fuse5_certain + uncertain_feature5)
        sideout5 = self.sideout5(fuse5)

        certain_feature4 = F.upsample(F.sigmoid(sideout5),size=e4_rgb.size()[2:], mode='bilinear')*e4_rgb * e4_depth
        fuse4_conv1 = self.fuse4_conv1(e4_rgb+certain_feature4)
        fuse4_conv2 = self.fuse4_conv2(e4_depth+certain_feature4)
        fuse4_certain = self.fuse4_conv3(fuse4_conv1+fuse4_conv2)
        # uncertain_feature4_noconv = F.upsample(F.sigmoid(sideout5), size=e4_rgb.size()[2:], mode='bilinear') * torch.abs(e4_rgb - e4_depth)
        uncertain_feature4 = self.fuse4_conv4(F.upsample(F.sigmoid(sideout5),size=e4_rgb.size()[2:], mode='bilinear')*torch.abs(e4_rgb - e4_depth))
        fuse4 = self.fuse4_conv5(fuse4_certain + uncertain_feature4)
###
        fuse5_fpn = F.upsample(fuse5, size=fuse4.size()[2:], mode='bilinear')
        fpn_certain_feature4 = F.upsample(F.sigmoid(sideout5), size=fuse4.size()[2:], mode='bilinear') * fuse4 * fuse5_fpn
        fuse4_fpn_conv1 = self.fuse4_conv1_fpn(fuse5_fpn + fpn_certain_feature4)
        fuse4_fpn_conv2 = self.fuse4_conv2_fpn(fuse4 + fpn_certain_feature4)
        fuse4_certain_fpn = self.fuse4_conv3_fpn(fuse4_fpn_conv1 + fuse4_fpn_conv2)
        fpn_uncertain_feature4 = self.fuse4_conv4_fpn(
            F.upsample(F.sigmoid(sideout5), size=fuse4.size()[2:], mode='bilinear') * torch.abs(fuse4 - fuse5_fpn))
        fuse4_fpn = self.fuse4_conv5_fpn(fuse4_certain_fpn + fpn_uncertain_feature4)
        output4 = self.output4(fuse4_fpn)
        sideout4 = self.sideout4(output4)

        certain_feature3 = F.upsample(F.sigmoid(sideout4),size=e3_rgb.size()[2:], mode='bilinear')*e3_rgb * e3_depth
        fuse3_conv1 = self.fuse3_conv1(e3_rgb + certain_feature3)
        fuse3_conv2 = self.fuse3_conv2(e3_depth + certain_feature3)
        fuse3_certain = self.fuse3_conv3(fuse3_conv1 + fuse3_conv2)
        # uncertain_feature3_noconv = F.upsample(F.sigmoid(sideout4), size=e3_rgb.size()[2:], mode='bilinear') * torch.abs(e3_rgb - e3_depth)
        uncertain_feature3 = self.fuse3_conv4( F.upsample(F.sigmoid(sideout4),size=e3_rgb.size()[2:], mode='bilinear')*torch.abs(e3_rgb - e3_depth))
        fuse3 = self.fuse3_conv5(fuse3_certain + uncertain_feature3)
        ##
        output4_fpn = F.upsample(output4, size=fuse3.size()[2:], mode='bilinear')
        fpn_certain_feature3 = F.upsample(F.sigmoid(sideout4), size=fuse3.size()[2:], mode='bilinear') * output4_fpn * fuse3
        fuse3_fpn_conv1 = self.fuse3_conv1_fpn(output4_fpn + fpn_certain_feature3)
        fuse3_fpn_conv2 = self.fuse3_conv2_fpn(fuse3 + fpn_certain_feature3)
        fuse3_certain_fpn = self.fuse3_conv3_fpn(fuse3_fpn_conv1 + fuse3_fpn_conv2)

        fpn_uncertain_feature3 = self.fuse3_conv4_fpn(
            F.upsample(F.sigmoid(sideout4), size=fuse3.size()[2:], mode='bilinear') * torch.abs(fuse3 - output4_fpn))
        fuse3_fpn = self.fuse3_conv5_fpn(fuse3_certain_fpn + fpn_uncertain_feature3)
        output3 = self.output3(fuse3_fpn)
        sideout3 = self.sideout3(output3)

        certain_feature2 = F.upsample(F.sigmoid(sideout3),size=e2_rgb.size()[2:], mode='bilinear')*e2_rgb* e2_depth
        fuse2_conv1 = self.fuse2_conv1(e2_rgb + certain_feature2)
        fuse2_conv2 = self.fuse2_conv2(e2_depth + certain_feature2)
        fuse2_certain = self.fuse2_conv3(fuse2_conv1 + fuse2_conv2)
        # uncertain_feature2_noconv = F.upsample(F.sigmoid(sideout3), size=e2_rgb.size()[2:], mode='bilinear') * torch.abs(e2_rgb - e2_depth)
        uncertain_feature2 = self.fuse2_conv4(F.upsample(F.sigmoid(sideout3),size=e2_rgb.size()[2:], mode='bilinear')*torch.abs(e2_rgb - e2_depth))
        fuse2 = self.fuse2_conv5(fuse2_certain + uncertain_feature2)

        output3_fpn = F.upsample(output3, size=fuse2.size()[2:], mode='bilinear')
        fpn_certain_feature2 = F.upsample(F.sigmoid(sideout3), size=fuse2.size()[2:], mode='bilinear') * output3_fpn * fuse2
        fuse2_fpn_conv1 = self.fuse2_conv1_fpn(output3_fpn + fpn_certain_feature2)
        fuse2_fpn_conv2 = self.fuse2_conv2_fpn(fuse2 + fpn_certain_feature2)
        fuse2_certain_fpn = self.fuse2_conv3_fpn(fuse2_fpn_conv1 + fuse2_fpn_conv2)

        fpn_uncertain_feature2 = self.fuse2_conv4_fpn(
            F.upsample(F.sigmoid(sideout3), size=fuse2.size()[2:], mode='bilinear') * torch.abs(fuse2 - output3_fpn))
        fuse2_fpn = self.fuse2_conv5_fpn(fuse2_certain_fpn + fpn_uncertain_feature2)
        output2 = self.output2(fuse2_fpn)
        sideout2 = self.sideout2(output2)

        certain_feature1 = F.upsample(F.sigmoid(sideout2),size=e1_rgb.size()[2:],mode='bilinear')*e1_rgb * e1_depth
        fuse1_conv1 = self.fuse1_conv1(e1_rgb+certain_feature1)
        fuse1_conv2 = self.fuse1_conv2(e1_depth+certain_feature1)
        fuse1_certain = self.fuse1_conv3(fuse1_conv1+fuse1_conv2)
        # uncertain_feature1_noconv = F.upsample(F.sigmoid(sideout2), size=e1_rgb.size()[2:], mode='bilinear') * torch.abs(e1_rgb - e1_depth)
        uncertain_feature1 = self.fuse1_conv4(F.upsample(F.sigmoid(sideout2),size=e1_rgb.size()[2:],mode='bilinear')*torch.abs(e1_rgb - e1_depth))
        fuse1 = self.fuse1_conv5(fuse1_certain+uncertain_feature1)

        output2_fpn = F.upsample(output2, size=fuse1.size()[2:], mode='bilinear')
        fpn_certain_feature1 = F.upsample(F.sigmoid(sideout2), size=fuse1.size()[2:],
                                          mode='bilinear') * output2_fpn * fuse1
        fuse1_fpn_conv1 = self.fuse1_conv1_fpn(output2_fpn + fpn_certain_feature1)
        fuse1_fpn_conv2 = self.fuse1_conv2_fpn(fuse1 + fpn_certain_feature1)
        fuse1_certain_fpn = self.fuse1_conv3_fpn(fuse1_fpn_conv1 + fuse1_fpn_conv2)

        fpn_uncertain_feature1 = self.fuse1_conv4_fpn(
            F.upsample(F.sigmoid(sideout2), size=fuse1.size()[2:], mode='bilinear') * torch.abs(fuse1 - output2_fpn))
        fuse1_fpn = self.fuse1_conv5_fpn(fuse1_certain_fpn + fpn_uncertain_feature1)
        output1 = self.output1(fuse1_fpn)

        if self.training:
            return sideout5,sideout4,sideout3,sideout2,output1
        return output1

if __name__ == "__main__":
    model = RGBD_sal()
    depth = torch.randn(1, 3, 256, 256)
    input = torch.randn(1, 3, 256, 256)
    flops, params = profile(model,inputs=(input,depth))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops,params)