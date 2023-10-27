import time
from options.train_options import TrainOptions
from models.networks import MISS, VGGLoss, load_checkpoint
from models.RAFD import RAFD
import torch.nn as nn
import os
import torch
from torch.utils.data import DataLoader
import cv2
from torchvision import utils
from util.util import generate_label_color, onechannel_to_one_hot
from tqdm import tqdm
import numpy as np

opt = TrainOptions().parse()

def CreateDataset(opt):
    from data.aligned_dataset_test import AlignedDataset
    dataset = AlignedDataset()
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

start_epoch, epoch_iter = 1, 0

train_data = CreateDataset(opt)

train_loader = DataLoader(
    train_data, batch_size=opt.batchSize, shuffle=False, num_workers=opt.nThreads, pin_memory=True
)
dataset_size = len(train_loader)
warp_model = RAFD(opt, 45)

warp_model.eval()
warp_model.cuda()
print("load warp model: ", opt.warp_checkpoint)
load_checkpoint(warp_model, opt.warp_checkpoint)


gen_model = MISS(29, 3)
gen_model.eval()
gen_model.cuda()
print("load gen model: ", opt.gen_checkpoint)
load_checkpoint(gen_model, opt.gen_checkpoint)

criterionL1 = nn.L1Loss()
criterionVGG = VGGLoss()
total_steps = (start_epoch - 1) * dataset_size + epoch_iter

tryon_path = os.path.join("result", "tryon", opt.name)
os.makedirs(tryon_path, exist_ok=True)
warp_path = os.path.join("result", "test_all", opt.name)
os.makedirs(warp_path, exist_ok=True)

step = 0
step_per_batch = dataset_size


for epoch in range(1, 2):
    for data in tqdm(train_loader):
        iter_start_time = time.time()

        iter_start_time = time.time()

        total_steps += 1
        epoch_iter += 1

        source_img = data["source_img"].cuda()
        source_cloth_img = data["source_cloth_img"].cuda()
        target_cloth_img = data["target_cloth_img"].cuda()
        source_pose = data["source_pose"].cuda()
        source_densepose = data["source_densepose"].cuda()
        source_densepose_forshow = data["source_densepose_forshow"].cuda()
        source_preserve_mask = data["source_preserve_mask"].cuda()
        source_cloth_mask = data["source_cloth_mask"].cuda()
        target_cloth_mask = data["target_cloth_mask"].cuda()
        source_preserve_img_face = data["source_preserve_img_face"].cuda()
        source_preserve_img_other = data["source_preserve_img_other"].cuda()
        source_preserve_img_hand = data["source_preserve_img_hand"].cuda()
        source_densepose_preserve_mask = data["source_densepose_preserve_mask"].cuda()
        source_parsing_one_hot = data["source_parsing_one_hot"].cuda()
        source_dense_armhand_3channel = data["source_dense_armhand_3channel"].cuda()
        source_arm_mask = data["source_arm_mask"].cuda()
        source_tosor_mask_tensor = data["source_tosor_mask_tensor"].cuda()
        source_tosor_img = data["source_tosor_img"].cuda()
        source_face_mask = data["source_face_mask"].cuda()
        source_other_clothes_mask = data["source_other_clothes_mask"].cuda()
        source_background_img = data["source_background_img"].cuda()
        source_img_agnostic = data["source_img_agnostic"].cuda()
        source_img_agnostic = source_img_agnostic * (1 - source_cloth_mask)
        source_face_lower_mask = source_face_mask + source_other_clothes_mask

        concat = torch.cat([source_preserve_mask, source_densepose, source_pose], 1)

        with torch.no_grad():
            (
                flows,
                delta_list,
                img_all,
                edge_all,
                _,
                grid_x_all,
                grid_y_all,
                arm_mask_list,
            ) = warp_model(target_cloth_img, concat.cuda(), target_cloth_mask, iters=10)

        warped_cloth = img_all[-1]
        warped_prod_edge = edge_all[-1]
        fake_arm_parsing_3lableonehot = arm_mask_list[-1]
        fake_arm_parsing_mask = torch.argmax(
            fake_arm_parsing_3lableonehot, dim=1, keepdim=False
        ).float()
        fake_arm_parsing_mask_without_cloth = torch.where(
            fake_arm_parsing_mask == 4,
            torch.zeros_like(fake_arm_parsing_mask),
            fake_arm_parsing_mask,
        )
        fake_arm_parsing_mask_without_cloth = onechannel_to_one_hot(
            fake_arm_parsing_mask_without_cloth, 4
        )
        fake_arm_parsing_mask = onechannel_to_one_hot(fake_arm_parsing_mask, 5)

        added_image = (
            source_img_agnostic * (1 - warped_prod_edge)
            + warped_cloth * warped_prod_edge
        )
        preserved_skin_mask = (
            fake_arm_parsing_mask[:, 1:2, :, :] * source_arm_mask
            + fake_arm_parsing_mask[:, 2:3, :, :] * source_arm_mask
            + fake_arm_parsing_mask[:, 3:4, :, :] * source_tosor_mask_tensor
        )
        preserved_skin_image = source_img * preserved_skin_mask
        added_image = added_image * (1 - preserved_skin_mask) + preserved_skin_image

        gen_inputs = torch.cat(
            [
                added_image,
                warped_prod_edge,
                fake_arm_parsing_mask_without_cloth,
                source_dense_armhand_3channel,
                source_pose,
            ],
            1,
        )
        with torch.no_grad():
            gen_output, attention_mask_list, content_list, outputs = gen_model(
                gen_inputs
            )

        # ---------------------- save results ------------------------ #

        res_tryon = gen_output[0:1, :, :, :]
        combine = res_tryon[0]
        cv_img = (combine.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2
        rgb = (cv_img * 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tryon_path + "/" + str(step) + ".jpg", bgr)

        step += 1
