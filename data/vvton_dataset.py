import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import torch
import json
import numpy as np
import os.path as osp
from PIL import ImageDraw
import torchvision.transforms as transforms
import torch.nn.functional as F
from glob import glob

import logging
logger = logging.getLogger("logger")
class AlignedDataset(BaseDataset):

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        self.center_crop = transforms.CenterCrop((self.fine_height, self.fine_width))
        self.rgb_norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.to_tensor_and_norm_rgb = transforms.Compose(
            [self.center_crop, transforms.ToTensor(), self.rgb_norm,]
        )
        self.to_tensor = transforms.Compose(
            [
                self.center_crop,
                transforms.ToTensor(),
            ]
        )
        # collect all the image files
        folder = f"{self.opt.datamode}_frames"
        videos_search = f"{self.root}/{folder}/*/"
        video_folders = sorted(glob(videos_search))
        self.num_videos = len(video_folders)
        start, end = 0, self.num_videos
        self.image_names = []
        self._video_start_indices = set()
        self.register_videos(video_folders, start, end)

        self.isfind = True

    def __getitem__(self, index):
        target_cloth_name = self.get_input_cloth_name(index)
        cloth_path = self.get_input_cloth_path(index)
        image_name = self.get_person_image_name(index)
        image_path = self.get_person_image_path(index)
        result = {
            "target_cloth_name": target_cloth_name,
            "target_cloth_path": cloth_path,
            "source_image_name": image_name,
            "source_image_path": image_path,
        }
        result.update(self.get_cloth_representation(index))
        result.update(self.get_person_representation(index))
        return result
        
    def get_cloth_representation(self, index):
        """
        call all cloth loaders
        :param index:
        :return: cloth, cloth_mask
        """
        target_cloth_path = self.get_input_cloth_path(index)
        if target_cloth_path == "NOT_FOUND":
            print(f"{target_cloth_path=} not found")
            return {"target_cloth": torch.zeros_like(self.get_person_image(index)[0]), "target_cloth_mask": torch.zeros(1, 256, 192)}
        target_cloth_tensor = self.get_input_cloth(target_cloth_path)
        target_cloth_mask_tensor = self.get_input_cloth_mask(target_cloth_path)
        target_cloth_tensor = target_cloth_tensor * target_cloth_mask_tensor
        return {"target_cloth": target_cloth_tensor, "target_cloth_mask": target_cloth_mask_tensor}
    
    def get_input_cloth_mask(self, cloth_path: str):
        """ Creates a mask directly from the input_cloth """
        cloth_mask_path = cloth_path.replace(".jpg", ".png").replace("clothes_person", "cloth_mask")
        cloth_mask = Image.open(cloth_mask_path)
        cloth_mask = self.to_tensor(cloth_mask)
        return cloth_mask
    
    def get_input_cloth(self, cloth_path):
        """
        Calls _get_input_cloth_path() for a file path, then opens that path as an image
        tensor
        """
        c = self.open_image_as_normed_tensor(cloth_path)
        return c
    
    def get_input_cloth_path(self, index):
        image_path = self.image_names[index]
        video_id = AlignedDataset.extract_video_id(image_path)
        # extract which frame we're on
        frame_word = AlignedDataset.extract_frame_substring(image_path)

        if not self.opt.is_train and self.opt.tryon_list:
            if self.opt.model == "warp":
                cloth_path = self.video_ids_to_cloth_paths[video_id]
                return cloth_path
            else:  # try on module
                cloth_folder = osp.join(self.opt.warp_cloth_dir, video_id)
                search = f"{cloth_folder}/*{frame_word}*"
                cloth_path_matches = sorted(glob(search))
                cloth_path = cloth_path_matches[0]
                return cloth_path
        else:  # handle specifically to VVT folder structure
            if self.opt.model == "warp":
                # Grep from VVT dataset folder structure
                path = osp.join(self.root, "clothes_person")
                keyword = "cloth_front"
            else:  # Try-on Module
                if self.opt.warp_cloth_dir is None:  # we provide warp-cloth train
                    path = osp.join(self.root, self.opt.datamode, "warp-cloth")
                else:  # user specifies their own warp-cloth
                    path = self.opt.warp_cloth_dir
                keyword = f"cloth_front*{frame_word}"
        return self.find_cloth_path_under_vvt_root(keyword, path, video_id)
    
    def get_person_representation(self, index):
        """
        get all person represetations
        :param index:
        :return:
        """
        ret = {}
        # person image

        source_img_tensor, prev_image_tensor = self.get_person_image(index)

        # load parsing image(onthot and visual)
        source_parsing_tensor, source_parsing_label, source_parsing_forshow= self.get_person_parsed(index)

        # load densepose image(onthot and visual)
        source_densepose_tensor, source_densepose_label, source_densepose_forshow= self.get_person_densepose(index)
        
        # load pose points(onthot and visual)
        source_pose_tensor, source_pose_forshow = self.get_person_pose(index)
        
        # get source preserve mask
        source_preserve_mask_tensor, source_preserve_mask_forshow = self.get_person_preserve_mask(source_parsing_label)
        
        # get source cloth image
        source_cloth_mask_tensor = torch.FloatTensor((source_parsing_label.numpy() == 5).astype(np.int)) + torch.FloatTensor(
            (source_parsing_label.numpy() == 6).astype(np.int)) + torch.FloatTensor((source_parsing_label.numpy() == 7).astype(np.int))
        source_cloth_img_tensor = source_img_tensor * source_cloth_mask_tensor

        ret.update(
            {
                "source_image": source_img_tensor,
                "prev_image": prev_image_tensor,
                "source_parsing": source_parsing_tensor,
                "source_parsing_forshow": source_parsing_forshow,
                "source_densepose": source_densepose_tensor,
                "source_densepose_forshow": source_densepose_forshow,
                "source_pose": source_pose_tensor,
                "source_pose_forshow": source_pose_forshow,
                "source_preserve_mask": source_preserve_mask_tensor,
                "source_preserve_mask_forshow": source_preserve_mask_forshow,
                "source_cloth": source_cloth_img_tensor,
                "source_cloth_mask": source_cloth_mask_tensor,
            }
        )
        return ret
    
    def get_person_image(self, index):
        """
        helper function to get the person image; not used as input to the network. used
        instead to form the other input
        :param index:
        :return:
        """
        # person image
        image_path = self.get_person_image_path(index)
        im = self.open_image_as_normed_tensor(image_path)
        try:
            prev_image_path = self.get_person_image_path(index - 1)
            prev_image = self.open_image_as_normed_tensor(prev_image_path)
        except:
            prev_image = torch.zeros_like(im)

        return im, prev_image

    def get_person_parsed(self, index):
        """ loads parsing image """
        img_path = self.get_person_image_path(index)
        source_parsing_path = img_path.replace("frames", "parsing").replace(".jpg", ".png")
        source_parsing_pil = Image.open(source_parsing_path)
        source_parsing_tensor = self.to_tensor(source_parsing_pil) * 255.0
        source_parsing_one_hot = torch.zeros(20, 256, 192)
        source_parsing_one_hot = source_parsing_one_hot.scatter_(0, source_parsing_tensor.long(), 1)
        source_parsing_forshow = source_parsing_tensor / 20.0
        return source_parsing_one_hot, source_parsing_tensor, source_parsing_forshow

    def get_person_densepose(self, index):
        """ loads parsing image """
        img_path = self.get_person_image_path(index)
        source_densepose_path = img_path.replace("frames", "densepose").replace(".png", "_densepose.npy")
        source_densepose_np = np.load(source_densepose_path).astype(np.float32)
        source_densepose_tensor = torch.from_numpy(source_densepose_np)
        source_densepose_one_hot = torch.zeros(25, 256, 192)
        source_densepose_one_hot = source_densepose_one_hot.scatter_(0, source_densepose_tensor.long(), 1)
        source_densepose_forshow = source_densepose_tensor / 25.0
        return source_densepose_one_hot, source_densepose_tensor, source_densepose_forshow

    def get_person_pose(self, index, patch_size=32, r=5):
        img_path = self.get_person_image_path(index)
        source_pose_path = img_path.replace("frames", "frames_keypoint").replace(".png", "_keypoints.json")
        with open(os.path.join(source_pose_path), 'r') as f:
            pose_label = json.load(f)
        try:
            pose_data = pose_label['people'][0]['pose_keypoints']
        except IndexError:
            pose_data = [0 for i in range(54)]
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)
        hand_patch_mask = torch.zeros(1, self.fine_height, self.fine_width)
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
            one_map = self.to_tensor_and_norm_rgb(one_map.convert('RGB'))
            pose_map[i] = one_map[0]
            # get hand patch mask:
            if i == 4 or i == 7:
                x_min = max(int(pointx) - patch_size // 2, 0)
                x_max = x_min + patch_size
                if x_max > self.fine_width:
                    x_max = self.fine_width
                    x_min = x_max - patch_size

                y_min = max(int(pointy) - patch_size // 2, 0)
                y_max = y_min + patch_size
                if y_max > self.fine_height:
                    y_max = self.fine_height
                    y_min = y_max - patch_size
                hand_patch_mask[:, y_min:y_max, x_min:x_max] = 1
                hand_patch_index = [y_min, y_max, x_min, x_max]
        source_pose_tensor = pose_map
        # just for visualization
        source_pose_tensor_forshow = self.to_tensor(im_pose)
        source_pose_tensor_forshow = torch.cat([source_pose_tensor_forshow, source_pose_tensor_forshow, source_pose_tensor_forshow], 0)
        return source_pose_tensor, source_pose_tensor_forshow

    def get_person_preserve_mask(self, source_parsing_tensor):
        source_face_mask_tensor = torch.FloatTensor(
            (source_parsing_tensor.numpy() == 1).astype(np.int) + (source_parsing_tensor.numpy() == 2).astype(np.int) + (source_parsing_tensor.numpy() == 13).astype(np.int))
        source_other_clothes_mask_tensor = torch.FloatTensor(
            (source_parsing_tensor.numpy() == 8).astype(np.int) + (source_parsing_tensor.numpy() == 9).astype(np.int) + (source_parsing_tensor.numpy() == 12).astype(np.int) + (
                    source_parsing_tensor.numpy() == 16).astype(np.int) + (source_parsing_tensor.numpy() == 17).astype(np.int) + (source_parsing_tensor.numpy() == 18).astype(
                np.int) + (source_parsing_tensor.numpy() == 19).astype(np.int))
        source_preserve_mask_tensor = torch.cat([source_face_mask_tensor, source_other_clothes_mask_tensor], 0)
        source_preserve_mask_forshow = source_face_mask_tensor + source_other_clothes_mask_tensor
        return source_preserve_mask_tensor, source_preserve_mask_forshow
    
    def open_image_as_normed_tensor(self, path):
        img = Image.open(path)
        tensor = self.to_tensor_and_norm_rgb(img)
        return tensor
    
    def find_cloth_path_under_vvt_root(self, keyword, path, video_id):
        # TODO FIX ME: for some reason fw_gan_vvt's clothes_persons folder is in upper
        #  case. this is a temporay hack; we should really lowercase those folders.
        #  it also removes the ending sub-id, which is the garment id
        video_id, cloth_id = video_id.lower().split("-")
        cloth_folder = osp.join(path, video_id)
        search = f"{cloth_folder}-{cloth_id}/{video_id}-{cloth_id}*{keyword}.*"
        cloth_path_matches = sorted(glob(search))
        if len(cloth_path_matches) == 0:
            logger.debug(
                f"{search=} not found, relaxing search to any cloth term. We should probably fix this later."
            )
            search = f"{cloth_folder}/{video_id}-{cloth_id}*cloth*"
            cloth_path_matches = sorted(glob(search))
            logger.debug(f"{search=} found {cloth_path_matches=}")
        if len(cloth_path_matches) == 0:
            return "NOT_FOUND"
        assert (
            len(cloth_path_matches) > 0
        ), f"{search=} not found. Try specifying --warp_cloth_dir"
        return cloth_path_matches[0]
    
    def get_input_cloth_name(self, index):
        cloth_path = self.get_input_cloth_path(index)
        if cloth_path == "NOT_FOUND":
            return "NOT_FOUND"
        if not self.opt.is_train and self.opt.tryon_list:
            video_id = AlignedDataset.extract_video_id(self.image_names[index])
        else:  # not specified, using VVT dataset folder structure
            video_id = AlignedDataset.extract_video_id(cloth_path)
        base_cloth_name = osp.basename(cloth_path)
        frame_name = osp.basename(self.get_person_image_name(index))
        # Cloth name belongs to a specific video.
        # e.g. 4he21d00f-g11/4he21d00f-g11@10=cloth_front.jpg
        name = osp.join(video_id, f"{base_cloth_name}.FOR.{frame_name}")
        return name
    
    
    def get_person_image_name(self, index):
        image_path = self.get_person_image_path(index)
        video_id = AlignedDataset.extract_video_id(image_path)
        name = osp.join(video_id, osp.basename(image_path))
        return name
    
    def get_person_image_path(self, index):
        # because we globbed, the path is the list
        return self.image_names[index]
    
    def _add_video_frames_to_image_names(self, video_folder):
        search = f"{video_folder}/*.png"
        frames = sorted(glob(search))
        self.image_names.extend(frames)

    def _record_video_start_index(self):
        # add the video index
        beg_index = len(self.image_names)
        self._video_start_indices.add(beg_index)
        
    def register_videos(self, video_folders, start=0, end=-1):
        """ Records what index each video starts at, and collects all the video frames
        in a flat list. """
        for video_folder in video_folders[start:end]:
            self._record_video_start_index()  # starts with 0
            self._add_video_frames_to_image_names(video_folder)
    @staticmethod
    def extract_frame_substring(path: str) -> str:
        """
        Assumes a path is formatted as **frame_NNN.extension. Extracts the "frame_NNN" part.
        """
        frame_keyword_start = path.find("frame_")
        frame_keyword_end = path.rfind(".")
        frame_word = path[frame_keyword_start:frame_keyword_end]
        return frame_word
    @staticmethod
    def extract_video_id(image_path):
        """
        Assumes the video ID is the folder that the file is immediately under .

        Example:
            path/to/FOLDER/file.ext -->
            returns "FOLDER"
        """
        return image_path.split(os.sep)[-2]
    
    def __len__(self):
        return len(self.image_names)

    def name(self):
        return 'AlignedDataset'

    