import datetime
import os
import torch
from torch import optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import utils_ssl.joint_transforms
from utils_ssl.datasets_stage2 import ImageFolder
from utils_ssl.misc import AvgMeter, check_mkdir
from model.model_stage1 import Crossmodal_Autoendoer
from model.model_stage2 import Contour_Estimation
from torch.backends import cudnn
from utils_downstream.ssim_loss import SSIM
import torch.nn as nn
import torch.nn.functional as F
cudnn.benchmark = True
torch.manual_seed(2018)
torch.cuda.set_device(0)

##########################hyperparameters###############################
ckpt_path = './saved_model'
exp_name = 'pretext_task2_stage2'
args = {
    'iter_num': 79900,  #50epoch
    'train_batch_size': 4,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'snapshot': ''
}
##########################data augmentation###############################
joint_transform = utils_ssl.joint_transforms.Compose([
    utils_ssl.joint_transforms.RandomCrop(256, 256),  # change to resize
    utils_ssl.joint_transforms.RandomHorizontallyFlip(),
    utils_ssl.joint_transforms.RandomRotate(10)
])
img_transform = transforms.Compose([
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
target_transform = transforms.ToTensor()
##########################################################################
image_root = ''
depth_root = ''
gt_root = ''

train_set = ImageFolder(image_root, gt_root,depth_root, joint_transform, img_transform, target_transform)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=0, shuffle=True)


criterion = nn.BCEWithLogitsLoss().cuda()
criterion_BCE = nn.BCELoss().cuda()
criterion_mse = nn.MSELoss().cuda()
criterion_mae = nn.L1Loss().cuda()
criterion_ssim = SSIM(window_size=11,size_average=True)
def ssimmae(pre,gt):
    maeloss = criterion_mae(pre,gt)
    ssimloss = 1-criterion_ssim(pre,gt)
    loss = ssimloss+maeloss
    return loss

log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    #############################ResNet pretrained###########################
    #res18[2,2,2,2],res34[3,4,6,3],res50[3,4,6,3],res101[3,4,23,3],res152[3,8,36,3]
    model_pretext1 = Crossmodal_Autoendoer()
    net_pretext1 = model_pretext1.cuda()
    net_pretext1.load_state_dict(torch.load(os.path.join('./saved_model/pretext_task1.pth')))
    net_pretext1.eval()

    model_pretext2 = Contour_Estimation()
    net = model_pretext2.cuda().train()


    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])
    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net_pretext1,net, optimizer)


#########################################################################

def train(net_pretext1,net, optimizer):
    curr_iter = args['last_iter']
    while True:
        total_loss_record, loss1_record, loss2_record, loss3_record, loss4_record, loss5_record, loss6_record, loss7_record, loss8_record,loss9_record,loss10_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']
            # data\binarizing\Variable
            images, depths, gts  = data
            gts[gts > 0.5] = 1
            gts[gts != 1] = 0
            batch_size = images.size(0)
            inputs = Variable(images).cuda()
            labels = Variable(gts).cuda()
            depths = Variable(depths).cuda()
            b, c, h, w = labels.size()
            optimizer.zero_grad()
            target_1 = F.upsample(labels, size=h // 2, mode='nearest')
            target_2 = F.upsample(labels,  size=h // 4, mode='nearest')
            target_3 = F.upsample(labels,  size=h // 8, mode='nearest')
            target_4 = F.upsample(labels,  size=h // 16, mode='nearest')

            ##########loss#############
            depth_3 = torch.cat((depths, depths, depths), 1)
            e5_rgb,e4_rgb,e3_rgb,e2_rgb,e1_rgb, e5_depth,e4_depth,e3_depth,e2_depth,e1_depth = net_pretext1(
                inputs, depth_3)  # hed
            sideout5, sideout4, sideout3, sideout2, output1 = net(e5_rgb, e4_rgb, e3_rgb, e2_rgb, e1_rgb, e5_depth, e4_depth, e3_depth, e2_depth, e1_depth)
            loss1 = criterion_mae(F.sigmoid(sideout5), target_4)
            loss2 = criterion_mae(F.sigmoid(sideout4), target_3)
            loss3 = criterion_mae(F.sigmoid(sideout3), target_2)
            loss4 = criterion_mae(F.sigmoid(sideout2), target_1)
            loss5 = criterion_mae(F.sigmoid(output1), labels)

            total_loss = loss1 + loss2 + loss3 + loss4 + loss5
            total_loss.backward()
            optimizer.step()
            total_loss_record.update(total_loss.item(), batch_size)
            loss1_record.update(loss1.item(), batch_size)
            loss2_record.update(loss2.item(), batch_size)
            loss3_record.update(loss3.item(), batch_size)
            loss4_record.update(loss4.item(), batch_size)
            loss5_record.update(loss5.item(), batch_size)

            #############log###############
            curr_iter += 1
            log = '[iter %d], [total loss %.5f],[loss4 %.5f],[loss5 %.5f],[lr %.13f] ' % \
                  (curr_iter, total_loss_record.avg, loss4_record.avg, loss5_record.avg, optimizer.param_groups[1]['lr'])
            print(log)
            open(log_path, 'a').write(log + '\n')
            if curr_iter == args['iter_num']:
                torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
                return
    ###############end###############


if __name__ == '__main__':
    main()
