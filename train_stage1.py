import time

from options.train_options import TrainOptions
from models.networks import VGGLoss, save_checkpoint
from models.RAFD import RAFD
import torch.nn as nn
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime
from torchvision import utils
from util.util import generate_label_color

opt = TrainOptions().parse()
def CreateDataset(opt):
    from data.vvton_dataset import AlignedDataset
    dataset = AlignedDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

def cat2rgb(tensor):
    return torch.cat([tensor,tensor,tensor],1)
def getshow(tensor, lable_num):
    return generate_label_color(tensor, lable_num)[0].unsqueeze(0).cuda().float()

start_epoch, epoch_iter = 1, 0

train_data = CreateDataset(opt)

train_loader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=True,
                                               num_workers=8, pin_memory=True)
dataset_size = len(train_loader)
print('#training images = %d' % dataset_size)

warp_model = RAFD(opt, 45)
warp_model.train()
warp_model.cuda()


criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss()

params_warp = [p for p in warp_model.parameters()]
optimizer_warp = torch.optim.AdamW(params_warp, lr=0.0002, betas=(0.5, 0.999))

total_steps = (start_epoch - 1) * dataset_size + epoch_iter
step = 0
step_per_batch = dataset_size

log_path = os.path.join(opt.log_dir, opt.name, 'log')
os.makedirs(log_path,exist_ok=True)
writer = SummaryWriter(log_path)

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size

    for i, data in enumerate(train_loader):

        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1

        source_image = data['source_image'].cuda()
        source_parsing = data['source_parsing'].cuda()
        source_parsing_forshow = data['source_parsing_forshow'].cuda()
        source_densepose = data['source_densepose'].cuda()
        source_densepose_forshow = data['source_densepose_forshow'].cuda()
        source_pose = data['source_pose'].cuda()
        source_pose_forshow = data['source_pose_forshow'].cuda()
        target_cloth = data['target_cloth'].cuda()
        target_cloth_mask = data['target_cloth_mask'].cuda()
        source_cloth = data['source_cloth'].cuda()
        source_cloth_mask = data['source_cloth_mask'].cuda()
        source_preserve_mask = data['source_preserve_mask'].cuda()
        source_preserve_mask_forshow = data['source_preserve_mask_forshow'].cuda()

        concat = torch.cat([source_preserve_mask, source_densepose, source_pose], 1)

        iters = 5
        flows, delta_list, img_all, edge_all, _, grid_x_all, grid_y_all, _ = warp_model(target_cloth, concat.cuda(), target_cloth_mask, iters=iters)

        warped_cloth = img_all[-1]
        warped_prod_edge = edge_all[-1]
        epsilon = 0.001

        loss_all = 0
        loss_gamma = 0.9
        for num in range(iters):
            loss_l1 = criterionL1(img_all[num], source_cloth.cuda())
            loss_vgg = criterionVGG(img_all[num], source_cloth.cuda())
            loss_edge = criterionL1(edge_all[num], source_cloth_mask.cuda())
            b,c,h,w = grid_x_all[num].shape
            loss_flow_x = (grid_x_all[num].pow(2)+ epsilon*epsilon).pow(0.45)
            loss_flow_x = torch.sum(loss_flow_x)/(b*c*h*w)
            loss_flow_y = (grid_y_all[num].pow(2)+ epsilon*epsilon).pow(0.45)
            loss_flow_y = torch.sum(loss_flow_y)/(b*c*h*w)
            loss_second_smooth = loss_flow_x + loss_flow_y

            loss_all = loss_all + (num+1) * loss_l1 + (num + 1) * 0.2 * loss_vgg + (num+1) * 2 * loss_edge + (num + 1) * 0.1 * loss_second_smooth


        writer.add_scalars('loss_all', {'loss_l1': loss_l1, 'loss_second_smooth': loss_second_smooth,
                                        'loss_edge': loss_edge, 'loss_vgg': loss_vgg},
                            global_step=step)

        optimizer_warp.zero_grad()
        loss_all.backward()
        optimizer_warp.step()
        ############## Display results and errors ##########

        vis_path = os.path.join(log_path,'visuals')
        os.makedirs(vis_path, exist_ok=True)
        if step % 100 == 0:
            combine = torch.cat([
                source_image[0:1, :, :, :],
                getshow(source_parsing, 20),
                getshow(source_densepose, 25),
                source_pose_forshow[0:1, :, :, :],
                source_cloth[0:1, :, :, :],
                cat2rgb(source_cloth_mask[0:1, :, :, :]),
                cat2rgb(source_preserve_mask_forshow[0:1, :, :, :]),
                target_cloth[0:1, :, :, :],
                cat2rgb(target_cloth_mask[0:1, :, :, :]),
                warped_cloth[0:1, :, :, :],
                cat2rgb(warped_prod_edge[0:1, :, :, :]),
                ], 0).squeeze()

            utils.save_image(combine, os.path.join(vis_path, str(step) + '.jpg'), nrow=int(4), normalize=True, range=(-1, 1), )
        step += 1
        iter_end_time = time.time()
        iter_delta_time = iter_end_time - iter_start_time
        step_delta = (step_per_batch - step % step_per_batch) + step_per_batch * (opt.niter + opt.niter_decay - epoch)
        eta = iter_delta_time * step_delta
        eta = str(datetime.timedelta(seconds=int(eta)))
        time_stamp = datetime.datetime.now()
        now = time_stamp.strftime('%Y.%m.%d-%H:%M:%S')
        if step % 100 == 0:
            if opt.local_rank == 0:
                print('{}:{}:[step-{}]--[loss-{:.6f}]--[ETA-{}]'.format(now, epoch_iter, step, loss_all, eta))

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
            (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ### save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            save_path = os.path.join(opt.log_dir, opt.name, 'checkpoints')
            os.makedirs(save_path,exist_ok=True)
            save_checkpoint(warp_model,
                            os.path.join(opt.checkpoints_dir, opt.name, 'PBAFN_warp_epoch_%03d.pth' % (epoch + 1)))

    if epoch > opt.niter:
        warp_model.update_learning_rate(optimizer_warp)
