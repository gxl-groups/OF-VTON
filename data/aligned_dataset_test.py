import os.path
from data.base_dataset import BaseDataset
from PIL import Image
import torch
import json
import numpy as np
import os.path as osp
from PIL import ImageDraw
import torchvision.transforms as transforms
class AlignedDataset(BaseDataset):
    def initialize(self, opt):

        self.opt = opt
        self.root = opt.dataroot
        self.person_names = []
        self.cloth_names = []
        opt.phase = 'test'
        with open(os.path.join(self.root, 'test_pairs.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip()
                A, B = line.split(' ')
                self.person_names.append(A)
                self.cloth_names.append(B)
        self.transform_1 = transforms.Compose([transforms.ToTensor()])
        self.transform_2 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.fine_height = 256
        self.fine_width = 192
        self.radius = 5
    def __getitem__(self, index):

        person_name = self.person_names[index]
        cloth_name = self.cloth_names[index]
        source_parsing_path = os.path.join(self.root, 'test_label_cihp', person_name.replace('jpg', 'png'))
        source_parsing_pil = Image.open(source_parsing_path).convert('L')
        source_parsing_tensor = self.transform_1(source_parsing_pil) * 255.0
        source_parsing_one_hot = torch.zeros(20, 256, 192)
        source_parsing_one_hot = source_parsing_one_hot.scatter_(0, source_parsing_tensor.long(), 1)
        source_img_path = os.path.join(self.root, 'test_img', person_name)
        source_img_pil = Image.open(source_img_path).convert('RGB')
        source_img_tensor = self.transform_2(source_img_pil)
        target_cloth_img_path = os.path.join(self.root, 'test_color', cloth_name)
        target_cloth_img_pil = Image.open(target_cloth_img_path).convert('RGB')
        target_cloth_img_tensor = self.transform_2(target_cloth_img_pil)
        target_cloth_img_tensor_nomask = target_cloth_img_tensor
        target_cloth_mask_path = os.path.join(self.root, 'test_edge_GrabCut', cloth_name.replace('jpg', 'png'))
        target_cloth_mask_pil = Image.open(target_cloth_mask_path).convert('L')
        target_cloth_mask_tensor = self.transform_1(target_cloth_mask_pil)
        target_cloth_mask_tensor = torch.FloatTensor((target_cloth_mask_tensor.numpy() > 0.5).astype(np.int))
        target_cloth_img_tensor = target_cloth_img_tensor * target_cloth_mask_tensor
        source_pose_path = os.path.join(self.root, 'test_pose', person_name.replace('.jpg', '_keypoints.json'))
        with open(osp.join(source_pose_path), 'r') as f:
            pose_label = json.load(f)
            try:
                pose_data = pose_label['people'][0]['pose_keypoints']
            except IndexError:
                pose_data = [0 for i in range(54)]
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape((-1, 3))
        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        r = self.radius
        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)
        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]
            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
                pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
            one_map = self.transform_2(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
        source_pose_tensor = pose_map
        # just for visualization
        source_pose_tensor_forshow = self.transform_1(im_pose)
        source_densepose_path = os.path.join(self.root, 'test_densepose_new', person_name.replace('jpg', 'npy'))
        source_densepose_np = np.load(source_densepose_path).astype(np.float32)
        source_densepose_tensor = torch.from_numpy(source_densepose_np).unsqueeze(0)
        source_cloth_mask_tensor = torch.FloatTensor((source_parsing_tensor.numpy() == 5).astype(np.int)) + torch.FloatTensor(
            (source_parsing_tensor.numpy() == 6).astype(np.int)) + torch.FloatTensor((source_parsing_tensor.numpy() == 7).astype(np.int))
        source_cloth_img_tensor = source_img_tensor * source_cloth_mask_tensor
        source_cloth_img_tensor_nomask = source_img_tensor * source_cloth_mask_tensor + torch.ones_like(source_img_tensor) * (1 - source_cloth_mask_tensor)
        source_densepose_one_hot = torch.zeros(25, 256, 192)
        source_densepose_one_hot = source_densepose_one_hot.scatter_(0, source_densepose_tensor.long(), 1)
        source_densepose_forshow = source_densepose_tensor / 24
        source_densepose_arm_cat_tensor = torch.cat([source_densepose_one_hot[3].unsqueeze(0), source_densepose_one_hot[4].unsqueeze(0), source_densepose_one_hot[15].unsqueeze(0),
                                                     source_densepose_one_hot[16].unsqueeze(0), source_densepose_one_hot[17].unsqueeze(0),
                                                     source_densepose_one_hot[18].unsqueeze(0), source_densepose_one_hot[19].unsqueeze(0),
                                                     source_densepose_one_hot[20].unsqueeze(0), source_densepose_one_hot[21].unsqueeze(0),
                                                     source_densepose_one_hot[22].unsqueeze(0)], 0)
        source_tosor_mask_tensor = torch.FloatTensor((source_parsing_tensor.numpy() == 10).astype(np.int))
        source_tosor_img_tensor = source_img_tensor * source_tosor_mask_tensor

        dense_preserve_mask_tensor = torch.FloatTensor((source_densepose_np == 15).astype(np.int)) + torch.FloatTensor(
            (source_densepose_np == 16).astype(np.int)) + torch.FloatTensor((source_densepose_np == 17).astype(np.int)) + torch.FloatTensor(
            (source_densepose_np == 18).astype(np.int)) + torch.FloatTensor((source_densepose_np == 19).astype(np.int)) + torch.FloatTensor(
            (source_densepose_np == 20).astype(np.int)) + torch.FloatTensor((source_densepose_np == 21).astype(np.int)) + torch.FloatTensor(
            (source_densepose_np == 22).astype(np.int)).unsqueeze(0)


        # ver2.0: 2023.3.30加了hand，之前没加;之前没背景，现在加了背景
        dense_preserve_mask_left_tensor = torch.FloatTensor((source_densepose_np == 4).astype(np.int)) + torch.FloatTensor((source_densepose_np == 15).astype(np.int)) + torch.FloatTensor(
            (source_densepose_np == 17).astype(np.int)) + torch.FloatTensor((source_densepose_np == 19).astype(np.int)) + torch.FloatTensor(
            (source_densepose_np == 21).astype(np.int)).unsqueeze(0)
        dense_preserve_mask_right_tensor = torch.FloatTensor((source_densepose_np == 3).astype(np.int)) + torch.FloatTensor((source_densepose_np == 16).astype(np.int)) + torch.FloatTensor(
            (source_densepose_np == 18).astype(np.int)) + torch.FloatTensor((source_densepose_np == 20).astype(np.int)) + torch.FloatTensor(
            (source_densepose_np == 22).astype(np.int)).unsqueeze(0)
        dense_preserve_mask_others_tensor = 1 - dense_preserve_mask_left_tensor - dense_preserve_mask_right_tensor
        dense_preserve_mask_3channel_tensor = torch.cat((dense_preserve_mask_others_tensor, dense_preserve_mask_left_tensor, dense_preserve_mask_right_tensor), 0)

        source_arm_mask = torch.FloatTensor((source_parsing_tensor.numpy() == 14).astype(np.int) + (source_parsing_tensor.numpy() == 15).astype(np.int))
        source_arm_img_tensor = source_img_tensor * source_arm_mask
        source_hand_mask = torch.FloatTensor((source_densepose_np == 3).astype(np.int) + (source_densepose_np == 4).astype(np.int))
        source_hand_mask_tensor = source_arm_mask * source_hand_mask

        source_arm_parsing_3label = torch.zeros(3, 256, 192)
        source_arm_parsing_3label[1] = source_parsing_one_hot[14]
        source_arm_parsing_3label[2] = source_parsing_one_hot[15]
        source_arm_parsing_3label[0] = torch.ones((256, 192)) - source_arm_mask

        source_arm_parsing_4label = torch.zeros(4, 256, 192)
        source_arm_parsing_4label[1] = source_parsing_one_hot[14]
        source_arm_parsing_4label[2] = source_parsing_one_hot[15]
        source_arm_parsing_4label[3] = source_parsing_one_hot[10]
        source_arm_parsing_4label[0] = torch.ones((256, 192)) - source_arm_mask - source_tosor_mask_tensor

        source_arm_parsing_5label = torch.zeros(5, 256, 192)
        source_arm_parsing_5label[1] = source_parsing_one_hot[14]
        source_arm_parsing_5label[2] = source_parsing_one_hot[15]
        source_arm_parsing_5label[3] = source_parsing_one_hot[10]
        source_arm_parsing_5label[4] = source_cloth_mask_tensor.squeeze(0)
        source_arm_parsing_5label[0] = torch.ones((256, 192)) - source_arm_mask - source_tosor_mask_tensor - source_cloth_mask_tensor

        source_arm_parsing_2label = torch.zeros(2, 256, 192)
        source_arm_parsing_2label[1] = source_arm_mask
        source_arm_parsing_2label[0] = torch.ones((256, 192)) - source_arm_mask

        source_face_mask_tensor = torch.FloatTensor(
            (source_parsing_tensor.numpy() == 1).astype(np.int) + (source_parsing_tensor.numpy() == 2).astype(np.int) + (source_parsing_tensor.numpy() == 13).astype(np.int))

        source_other_clothes_mask_tensor = torch.FloatTensor(
            (source_parsing_tensor.numpy() == 8).astype(np.int) + (source_parsing_tensor.numpy() == 9).astype(np.int) + (source_parsing_tensor.numpy() == 12).astype(np.int) + (
                    source_parsing_tensor.numpy() == 16).astype(np.int) + (source_parsing_tensor.numpy() == 17).astype(np.int) + (source_parsing_tensor.numpy() == 18).astype(
                np.int) + (source_parsing_tensor.numpy() == 19).astype(np.int))
        source_preserve_mask_tensor = torch.cat([source_face_mask_tensor, source_other_clothes_mask_tensor], 0)
        source_preserve_mask_forshow = source_face_mask_tensor + source_other_clothes_mask_tensor

        source_preserve_img_face = source_img_tensor * source_face_mask_tensor
        source_preserve_img_other = source_img_tensor * source_other_clothes_mask_tensor
        source_preserve_img_hand = source_img_tensor * source_hand_mask_tensor

        source_tosor_mask_tensor = torch.FloatTensor((source_parsing_tensor.numpy() == 10).astype(np.int))
        source_tosor_img_tensor = source_img_tensor * source_tosor_mask_tensor

        source_background_mask = torch.FloatTensor((source_parsing_tensor.numpy() == 0).astype(np.int))
        source_background_img = source_img_tensor * source_background_mask

        source_img_agnostic_pil = self.get_img_agnostic(source_img_pil, source_parsing_pil, pose_data)
        source_img_agnostic_tensor = self.transform_2(source_img_agnostic_pil)
        source_img_agnostic_tensor = source_img_agnostic_tensor * (1 - source_arm_mask)
        source_img_agnostic_tensor = source_img_agnostic_tensor * (1 - source_face_mask_tensor) + source_preserve_img_face
        source_img_agnostic_tensor = source_img_agnostic_tensor * (1 - source_other_clothes_mask_tensor) + source_preserve_img_other
        source_img_agnostic_tensor = source_img_agnostic_tensor * (1 - source_hand_mask_tensor) + source_preserve_img_hand


        # set label 14 15 10 5 6 7 to 0
        source_parsing_cloth_agnostic_label = source_parsing_tensor.clone()
        source_parsing_cloth_agnostic_label[source_parsing_cloth_agnostic_label == 14] = 0
        source_parsing_cloth_agnostic_label[source_parsing_cloth_agnostic_label == 15] = 0
        source_parsing_cloth_agnostic_label[source_parsing_cloth_agnostic_label == 10] = 0
        source_parsing_cloth_agnostic_label[source_parsing_cloth_agnostic_label == 5] = 0
        source_parsing_cloth_agnostic_label[source_parsing_cloth_agnostic_label == 6] = 0
        source_parsing_cloth_agnostic_label[source_parsing_cloth_agnostic_label == 7] = 0

        source_parsing_cloth_agnostic_onehot = torch.zeros(20, 256, 192)
        source_parsing_cloth_agnostic_onehot = source_parsing_cloth_agnostic_onehot.scatter_(0, source_parsing_cloth_agnostic_label.long(), 1)

        if self.opt.isTrain:
            input_dict = {'source_parsing': source_parsing_tensor, 'source_parsing_cloth_agnostic_onehot':source_parsing_cloth_agnostic_onehot,
                          'source_parsing_cloth_agnostic_label': source_parsing_cloth_agnostic_label,
                          'source_img': source_img_tensor, 'source_parsing_path': source_parsing_path, 'source_img_path': source_img_path,
                          'target_cloth_img_path': target_cloth_img_path, # 'target_cloth_img_un_path': target_cloth_img_un_path,
                          'target_cloth_mask': target_cloth_mask_tensor, 'target_cloth_img': target_cloth_img_tensor, # 'target_cloth_mask_un': target_cloth_mask_un_tensor,
                          # 'target_cloth_img_un': target_cloth_img_un_tensor,
                          'source_pose_forshow': source_pose_tensor_forshow, 'source_pose': source_pose_tensor, 'source_densepose': source_densepose_one_hot,
                          'source_preserve_mask': source_preserve_mask_tensor, 'source_cloth_img': source_cloth_img_tensor, 'source_densepose_forshow': source_densepose_forshow,
                          'target_cloth_mask': target_cloth_mask_tensor, 'source_cloth_mask': source_cloth_mask_tensor, 'target_cloth_name': cloth_name,
                          'source_preserve_mask_forshow': source_preserve_mask_forshow, 'source_preserve_img_face': source_preserve_img_face,
                          'source_preserve_img_other': source_preserve_img_other, 'source_preserve_img_hand': source_preserve_img_hand,
                          'source_densepose_preserve_mask': dense_preserve_mask_tensor, 'target_cloth_img_nomask': target_cloth_img_tensor_nomask,
                          'source_cloth_img_nomask': source_cloth_img_tensor_nomask, 'source_arm_img': source_arm_img_tensor, 'source_tosor_img': source_tosor_img_tensor,
                          'source_parsing_one_hot': source_parsing_one_hot, 'source_hand_mask': source_hand_mask_tensor,
                          'source_dense_armhand_3channel': dense_preserve_mask_3channel_tensor,
                          'person_name': person_name, 'source_arm_parsing_2label': source_arm_parsing_2label,
                          'source_arm_parsing_3label': source_arm_parsing_3label, 'source_tosor_img_tensor': source_tosor_img_tensor, 'source_background_img': source_background_img,
                          'source_arm_parsing_4label': source_arm_parsing_4label, 'source_face_mask': source_face_mask_tensor, 'source_other_clothes_mask': source_other_clothes_mask_tensor,
                          'source_img_agnostic': source_img_agnostic_tensor, 'source_arm_parsing_5label': source_arm_parsing_5label, 'source_densepose_arm_cat': source_densepose_arm_cat_tensor,
                          'source_dense_armhand_3channel': dense_preserve_mask_3channel_tensor, 'source_tosor_mask_tensor': source_tosor_mask_tensor,
                          'source_arm_mask': source_arm_mask}

        return input_dict

    def __len__(self):
        return len(self.person_names) // (self.opt.batchSize * self.opt.num_gpus) * (self.opt.batchSize * self.opt.num_gpus)

    def name(self):
        return 'AlignedDataset'

    def get_img_agnostic(self, source_img_pil, source_parsing_pil, pose_data):
        agnostic = source_img_pil.copy()
        agnostic_draw = ImageDraw.Draw(agnostic)
        length_a = np.linalg.norm(pose_data[5] - pose_data[2])  # LShoulder - RShoulder
        length_b = np.linalg.norm(pose_data[11] - pose_data[8])  # Lhip-Rhip

        point = (pose_data[8] + pose_data[11]) / 2
        pose_data[8] = point + (pose_data[8] - point) / length_b * length_a
        pose_data[11] = point + (pose_data[11] - point) / length_b * length_a

        r = 5
        # mask arms
        agnostic_draw.line([tuple(pose_data[i, :2]) for i in [2, 5]], 'gray', width=r * 10, joint='curve')  #
        for i in [2, 5]:
            pointx, pointy = pose_data[i, 0], pose_data[i, 1]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')

        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue
            agnostic_draw.line([tuple(pose_data[j, :2]) for j in [i - 1, i]], 'gray', width=r * 10)
            pointx, pointy = pose_data[i, 0], pose_data[i, 1]
            agnostic_draw.ellipse((pointx - r * 5, pointy - r * 5, pointx + r * 5, pointy + r * 5), 'gray', 'gray')

        for i in [8, 11]:
            pointx, pointy = pose_data[i, 0], pose_data[i, 1]
            agnostic_draw.ellipse((pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6), 'gray', 'gray')
        agnostic_draw.line([tuple(pose_data[i, :2]) for i in [2, 8]], 'gray', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i, :2]) for i in [5, 11]], 'gray', width=r * 6)
        agnostic_draw.line([tuple(pose_data[i, :2]) for i in [8, 11]], 'gray', width=r * 12)
        agnostic_draw.polygon([tuple(pose_data[i, :2]) for i in [2, 5, 11, 8]], 'gray', 'gray')


        # mask neck
        pointx, pointy = pose_data[1, 0], pose_data[1, 1]
        agnostic_draw.rectangle((pointx - r * 7, pointy - r * 7, pointx + r * 7, pointy + r * 7), 'gray', 'gray')

        # mask arm
        agnostic = agnostic
        return agnostic
