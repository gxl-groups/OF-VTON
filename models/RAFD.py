import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.update import GMAUpdateBlock
from models.extractor import BasicEncoder
from models.corr import CorrBlock
from util.util import bilinear_sampler, coords_grid, upflow8
from options.train_options import TrainOptions
from models.att import Attention
opt = TrainOptions().parse()
try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    print('amp not available')
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

def apply_offset(offset):
    sizes = list(offset.size()[2:])
    grid_list = torch.meshgrid([torch.arange(size, device=offset.device) for size in sizes])
    grid_list = reversed(grid_list)
    # apply offset
    grid_list = [grid.float().unsqueeze(0) + offset[:, dim, ...] for dim, grid in enumerate(grid_list)]
    # normalize
    grid_list = [grid / ((size - 1.0) / 2.0) - 1.0 for grid, size in zip(grid_list, reversed(sizes))]

    return torch.stack(grid_list, dim=-1)


class RAFD(nn.Module):
    def __init__(self, args, input_channels):
        super(RAFD, self).__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        print('corr_radius: ', args.corr_radius)

        self.old_lr = args.lr
        self.old_lr_warp = args.lr * 0.2

        if 'dropout' not in self.args:
            self.args.dropout = 0
            print('no dropout')



        # feature network, context network, and update block

        self.fnet = BasicEncoder(input_dim=3, output_dim=256, norm_fn='batch', dropout=args.dropout)
        self.fnet2 = BasicEncoder(input_dim=input_channels, output_dim=256, norm_fn='batch', dropout=args.dropout)
        self.cnet = BasicEncoder(input_dim=3, output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = GMAUpdateBlock(self.args, hidden_dim=hdim)
        self.att = Attention(args=self.args, dim=cdim, heads=self.args.num_heads, max_pos_size=160, dim_head=cdim)

        netMain_layer1 = torch.nn.Sequential(torch.nn.Conv2d(2 * 256, out_channels=128, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1), torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
            torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1))
        self.netMain = []
        self.netMain.append(netMain_layer1)
        # self.netMain.append(netMain_layer2)
        self.netMain = nn.ModuleList(self.netMain)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                print('freeze bn')

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, edge, iters=5, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        grid_x_all = []
        grid_y_all = []
        filter_x = [[0, 0, 0], [1, -2, 1], [0, 0, 0]]
        filter_y = [[0, 1, 0], [0, -2, 0], [0, 1, 0]]
        filter_diag1 = [[1, 0, 0], [0, -2, 0], [0, 0, 1]]
        filter_diag2 = [[0, 0, 1], [0, -2, 0], [1, 0, 0]]
        weight_array = np.ones([3, 3, 1, 4])
        weight_array[:, :, 0, 0] = filter_x
        weight_array[:, :, 0, 1] = filter_y
        weight_array[:, :, 0, 2] = filter_diag1
        weight_array[:, :, 0, 3] = filter_diag2

        weight_array = torch.cuda.FloatTensor(weight_array).permute(3, 2, 0, 1)
        self.weight = nn.Parameter(data=weight_array, requires_grad=False)

        hdim = self.hidden_dim
        cdim = self.context_dim

        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.fnet(image1)
            fmap2 = self.fnet2(image2)

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)  # [N, 128+128, H/8, W/8]
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)  # [N, 128, H/8, W/8], [N, 128, H/8, W/8]
            net = torch.tanh(net)  # [N, 128, H/8, W/8]
            inp = torch.relu(inp)  # [N, 128, H/8, W/8]
            # attention, att_c, att_p = self.att(inp)
            attention = self.att(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        delta_flows = []
        img_all = []
        edge_all = []
        arm_mask_list = []
        grid_list = []
        lastFlow = None
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow, armmask = self.update_block(net, inp, corr, flow, attention)
            arm_mask_list.append(armmask)
            coords1 = coords1 + delta_flow
            delta_flows.append(delta_flow)

            down_flow = coords1 - coords0
            grid_down = apply_offset(down_flow)
            warped_feature1 = F.grid_sample(fmap1, grid_down, mode='bilinear', padding_mode='border')

            delta_flow = self.netMain[0](torch.cat([warped_feature1, fmap2], 1))
            delta_flows.append(delta_flow)
            coords1 = coords1 + delta_flow

            # upsample flow
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

            grid_up = apply_offset(flow_up)
            grid_list.append(grid_up.permute(0,3,1,2))
            img_all.append(F.grid_sample(image1.clone(), grid_up, mode='bilinear', padding_mode='border'))
            edge_all.append(F.grid_sample(edge.clone(), grid_up, mode='bilinear', padding_mode='zeros'))


            grid_x, grid_y = torch.split(grid_up.permute(0, 3, 1, 2), 1, dim=1)

            grid_x = F.conv2d(grid_x, self.weight)
            grid_y = F.conv2d(grid_y, self.weight)
            grid_x_all.append(grid_x)
            grid_y_all.append(grid_y)

        return flow_predictions, delta_flows, img_all, edge_all, grid_list, grid_x_all, grid_y_all, arm_mask_list

    def update_learning_rate(self, optimizer):
        lrd = opt.lr / opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def update_learning_rate_warp(self, optimizer):
        lrd = 0.2 * opt.lr / opt.niter_decay
        lr = self.old_lr_warp - lrd
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr_warp, lr))
        self.old_lr_warp = lr
