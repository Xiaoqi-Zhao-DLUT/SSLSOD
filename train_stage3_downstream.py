import torch
import torch.nn.functional as F
from torch.autograd import Variable
import  os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from datetime import datetime
from model.model_stage3 import RGBD_sal
from utils_downstream.dataset_rgbd_strategy2 import get_loader
from utils_downstream.utils import adjust_lr, AvgMeter
import torch.nn as nn
from torch.cuda import amp


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr_gen', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument('-beta1_gen', type=float, default=0.5,help='beta of Adam for generator')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight_decay')
parser.add_argument('--feat_channel', type=int, default=64, help='reduced channel of saliency feat')

opt = parser.parse_args()
print('Generator Learning Rate: {}'.format(opt.lr_gen))
# build models
generator = RGBD_sal()
generator.cuda()

pretrained_dict = torch.load(os.path.join('./saved_model/pretext_task1.pth'))
model_dict = generator.state_dict()
# print(pretrained_dict.items())
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
generator.load_state_dict(model_dict)

pretrained_dict = torch.load(os.path.join('./saved_model/pretext_task2.pth'))
model_dict = generator.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
print(pretrained_dict.items())
model_dict.update(pretrained_dict)
generator.load_state_dict(model_dict)

generator_params = generator.parameters()
generator_optimizer = torch.optim.Adam(generator_params, opt.lr_gen)

## load data
image_root = ''
depth_root = ''
gt_root = ''

train_loader = get_loader(image_root, gt_root,depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

## define loss

CE = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss(size_average=True, reduce=True)
# size_rates = [0.75,1,1.25]  # multi-scale training
size_rates = [1]  # multi-scale training
criterion = nn.BCEWithLogitsLoss().cuda()
criterion_mae = nn.L1Loss().cuda()
criterion_mse = nn.MSELoss().cuda()
use_fp16 = True
scaler = amp.GradScaler(enabled=use_fp16)

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))


    pred  = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou  = 1-(inter+1)/(union-inter+1)

    return (wbce+wiou).mean()



for epoch in range(1, opt.epoch+1):
    generator.train()
    loss_record = AvgMeter()
    print('Generator Learning Rate: {}'.format(generator_optimizer.param_groups[0]['lr']))
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            generator_optimizer.zero_grad()
            images, gts, depths = pack
            images = Variable(images)
            gts = Variable(gts)
            depths = Variable(depths)
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()
            # multi-scale training samples
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear',
                                          align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                # contours = F.upsample(contours, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                depths = F.upsample(depths, size=(trainsize, trainsize), mode='bilinear', align_corners=True)

            b, c, h, w = gts.size()
            target_1 = F.upsample(gts, size=h // 2, mode='nearest')
            target_2 = F.upsample(gts,  size=h // 4, mode='nearest')
            target_3 = F.upsample(gts,  size=h // 8, mode='nearest')
            target_4 = F.upsample(gts,  size=h // 16, mode='nearest')

            with amp.autocast(enabled=use_fp16):
                depth_3 = torch.cat((depths, depths, depths), 1)
                sideout5, sideout4, sideout3, sideout2, output1 = generator.forward(images, depth_3)  # hed
                loss1 = structure_loss(sideout5, target_4)
                loss2 = structure_loss(sideout4, target_3)
                loss3 = structure_loss(sideout3, target_2)
                loss4 = structure_loss(sideout2, target_1)
                loss5 = structure_loss(output1, gts)

                loss = loss1 + loss2 + loss3 + loss4 + loss5

            generator_optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(generator_optimizer)
            scaler.update()

            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)


        if i % 10 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], gen Loss: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_record.show()))
            # print(anneal_reg)


    adjust_lr(generator_optimizer, opt.lr_gen, epoch, opt.decay_rate, opt.decay_epoch)

    save_path = './saved_model/SSLSOD_v2'


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % opt.epoch == 0:
        torch.save(generator.state_dict(), save_path + 'Model' + '_%d' % epoch + '_gen.pth')
