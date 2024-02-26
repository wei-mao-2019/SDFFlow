import os
import torch
import numpy as np
from glob import glob
import torch.nn.functional as F
from tqdm import tqdm

import utils.general as utils
from utils import rend_util

class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 data_dir,
                 img_res,
                 scan_id=0,
                 **kargs
                 ):

        if not(type(scan_id) == str):
            self.instance_dir = os.path.join('/home/wei/Documents/projects/2022-nerf/SDFFlow/data', data_dir, 'scan{0}'.format(scan_id))
        else:
            self.instance_dir = os.path.join('/home/wei/Documents/projects/2022-nerf/SDFFlow/data', data_dir, '{0}'.format(scan_id))

        self.load_motion_mask = kargs.get('load_motion_mask', False)
        self.load_optical_flow = kargs.get('load_optical_flow', False)
        self.surrounding_pixels = kargs.get('surrounding_pixels', 0)
        self.edge_sampling = kargs.get('edge_sampling', True)
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        self.test_cam = ''

        assert os.path.exists(self.instance_dir), f"Data directory {self.instance_dir} is empty"

        self.sampling_idx = None

        image_dir = '{0}/image'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        self.n_images = len(image_paths)

        self.cam_file = '{0}/cameras.npz'.format(self.instance_dir)
        camera_dict = np.load(self.cam_file)
        self.n_cams = len(camera_dict.keys())//2
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_cams)]
        self.scale_mats = scale_mats
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_cams)]

        self.n_images = self.n_images//self.n_cams
        self.training_frames = kargs.get('training_frames', np.arange(self.n_images).tolist())

        self.intrinsics_all = []
        self.pose_all = []

        print('>>> load camera poses')
        for i, (scale_mat, world_mat) in enumerate(zip(scale_mats, world_mats)):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        self.rgb_images = {}
        self.fg = {}
        self.bg = {}
        self.img_keys = []
        print(f'>>> load images; training frames {self.training_frames}')
        for i,path in enumerate(image_paths):
            img_key = path.split('/')[-1][:-4]
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            # set background to be black
            rgb=torch.from_numpy(rgb).float()
            # if no motion mask then use color to find bg and fg
            if not self.load_motion_mask:
                self.fg[img_key] = torch.from_numpy(np.where(rgb.sum(axis=-1)<3.0)[0])
                self.bg[img_key] = torch.from_numpy(np.where(rgb.sum(axis=-1)>=3.0)[0])
                rgb[self.bg[img_key]] = 0.0
            self.rgb_images[img_key] = rgb

            # used to train on single frame or less frames.
            if int(img_key.split('_')[-1])-1 not in self.training_frames:
                continue
            if img_key.split('_')[0] == self.test_cam:
                continue
            self.img_keys.append(img_key)

        # load mask
        if self.load_motion_mask:
            print('>>> load motion mask')
            self.motion_mask = {}
            self.fg = {}
            self.bg = {}
            self.fg_edge = {}
            self.bg_edge = {}
            mask_dir = '{0}/motion_mask'.format(self.instance_dir)
            mask_paths = sorted(utils.glob_imgs(mask_dir))
            for i, path in enumerate(mask_paths):
                img_key = path.split('/')[-1][:-4]
                # if img_key not in self.img_keys:
                #     continue
                motion = rend_util.load_rgb(path) # motion seg is 0, bg is 1
                motion = (motion[0] < 0.5).reshape(-1).astype(np.float32)
                # motion = (motion.reshape(3, -1).transpose(1, 0).sum(axis=-1)<0.3).astype(np.float)
                self.motion_mask[img_key] = torch.from_numpy(motion).float()
                self.fg[img_key] = torch.from_numpy(np.where(motion)[0])
                self.bg[img_key] = torch.from_numpy(np.where(np.logical_not(motion))[0])
                self.rgb_images[img_key][self.bg[img_key]] = 0.0 # whether to use background prior

                #get edge pixels
                if self.edge_sampling:
                    motion_tmp = torch.from_numpy(motion.reshape(img_res))
                    ws = torch.ones(5, 5) / 25
                    out = F.conv2d(motion_tmp[None, None].float(), ws[None, None].float(), padding=(2,2))[0,0].reshape(-1)
                    is_edge = torch.logical_and(out>0,out<0.999)
                    self.fg_edge[img_key] = torch.where(torch.logical_and(is_edge,self.motion_mask[img_key]>0))[0]
                    self.bg_edge[img_key] = torch.where(torch.logical_and(is_edge,self.motion_mask[img_key]<0.999))[0]


        if self.load_optical_flow:
        # if False:
            print('>>> load optical flow')
            self.optical_flow = {}
            of_dir = '{0}/optical_flow'.format(self.instance_dir)
            of_paths = sorted(glob(of_dir+'/*.exr'))
            for i, path in enumerate(of_paths):
                img_key = path.split('/')[-1][:-4]
                # if img_key not in self.img_keys:
                #     continue
                # we assume (u0,v0)+OF to be the pixel location at next/previous frame
                flow_back, flow_forward = utils.exr2flow(path)
                self.optical_flow[img_key] = (torch.from_numpy(flow_back).reshape(-1,2) if (int(img_key.split('_')[1])-1) > 0 else None,
                                              torch.from_numpy(-flow_forward).reshape(-1,2) if (int(img_key.split('_')[1])-1) < (self.n_images-1) else None)
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        self.uv = uv.reshape(2, -1).transpose(1, 0)
        uv_list = self.uv.data.numpy().astype(np.int64).tolist()
        uv_list = [tuple(sub) for sub in uv_list]
        self.idx2uv = dict(enumerate(uv_list))
        self.uv2idx = {v:k for k,v in self.idx2uv.items()}
        self.idx = -1
        print('>>> data loaded')
        print(f'total loaded images: {self.__len__()}.')

    def __len__(self):
        # return self.n_images*self.n_cams
        return len(self.img_keys)

    def __getitem__(self, idx):
        if self.idx >= 0:
            idx = self.idx
        img_key = self.img_keys[idx]
        cam_idx = int(img_key.split('_')[0])
        img_idx = int(img_key.split('_')[1])-1

        sample = {
            "uv": self.uv,
            "intrinsics": self.intrinsics_all[cam_idx],
            "pose": self.pose_all[cam_idx],
            # if only one frame is trained then no flow needed
            "t": torch.tensor(img_idx) if len(self.training_frames) != 1 else torch.tensor(0)
        }
        ground_truth = {}
        if self.sampling_idx is None:
            ground_truth['rgb'] = self.rgb_images[img_key]
            if self.load_motion_mask:
                sample['motion_mask'] = self.motion_mask[img_key]
                sample['bg_mask'] = 1 - sample['motion_mask']

            if self.load_optical_flow:
                sample['optical_flow_back'] = self.optical_flow[img_key][0] if img_idx > 0 else torch.zeros(
                    self.total_pixels, 2)
                sample['optical_flow_forward'] = self.optical_flow[img_key][1] if img_idx < (
                            self.n_images - 1) else torch.zeros(self.total_pixels, 2)
                sample['pose_back'] = self.pose_all[cam_idx]
                sample['pose_forward'] = self.pose_all[cam_idx]
                sample['image'] = self.rgb_images[img_key].reshape(self.img_res + [3])
                sample['image_fwd'] = self.rgb_images[f"{cam_idx:03d}_{img_idx+2:04d}"].reshape(self.img_res + [3]) if img_idx < (
                            self.n_images - 1) else torch.zeros_like(self.rgb_images[img_key].reshape(self.img_res + [3]))
                sample['image_back'] = self.rgb_images[f"{cam_idx:03d}_{img_idx:04d}"].reshape(
                    self.img_res + [3]) if img_idx > 0 else torch.zeros_like(self.rgb_images[img_key].reshape(self.img_res + [3]))

        elif self.sampling_idx is not None:
            sampling_size = len(self.sampling_idx)
            num_bg = sampling_size//3
            num_fg = sampling_size - num_bg
            if self.edge_sampling:
                num_bg_edge = sampling_size//10
                num_fg_edge = sampling_size//10
                num_fg = sampling_size - num_bg - num_bg_edge - num_fg_edge
            bg_idx = torch.randperm(self.bg[img_key].shape[0])[:num_bg]
            fg_idx = torch.randperm(self.fg[img_key].shape[0])[:num_fg]
            if self.edge_sampling:
                bg_edge_idx = torch.randperm(self.bg_edge[img_key].shape[0])[:num_bg_edge]
                fg_edge_idx = torch.randperm(self.fg_edge[img_key].shape[0])[:num_fg_edge]
                sampling_idx = torch.cat([self.fg[img_key][fg_idx],self.fg_edge[img_key][fg_edge_idx],
                                          self.bg[img_key][bg_idx],self.bg_edge[img_key][bg_edge_idx]],dim=0)
                bg_mask = torch.zeros_like(sampling_idx)
                bg_mask[(num_fg+num_fg_edge):] = 1.0# if is 1 it is bg
            else:
                sampling_idx = torch.cat([self.fg[img_key][fg_idx],
                                          self.bg[img_key][bg_idx]], dim=0)
                bg_mask = torch.zeros_like(sampling_idx)
                bg_mask[num_fg:] = 1.0# if is 1 it is bg
            ground_truth["rgb"] = self.rgb_images[img_key][sampling_idx, :]
            sample["uv"] = self.uv[sampling_idx, :]
            sample['bg_mask'] = bg_mask

            if self.load_motion_mask:
                sample['motion_mask'] = self.motion_mask[img_key][sampling_idx]
            if self.load_optical_flow:
                sample['optical_flow_back'] = self.optical_flow[img_key][0][sampling_idx] if img_idx > 0 else torch.zeros(len(sampling_idx),2)
                sample['optical_flow_forward'] = self.optical_flow[img_key][1][sampling_idx] if img_idx < (self.n_images-1) else torch.zeros(len(sampling_idx),2)
                sample['pose_back'] = self.pose_all[cam_idx] if img_idx > 0 else torch.zeros(len(sampling_idx),2)
                sample['pose_forward'] = self.pose_all[cam_idx] if img_idx < (self.n_images-1) else torch.zeros(len(sampling_idx),2)
                sample['image'] = self.rgb_images[img_key].reshape(self.img_res + [3])
                sample['image_fwd'] = self.rgb_images[f"{cam_idx:03d}_{img_idx + 2:04d}"].reshape(
                    self.img_res + [3]) if img_idx < (self.n_images - 1) else torch.zeros_like(
                    self.rgb_images[img_key].reshape(self.img_res + [3]))
                sample['image_back'] = self.rgb_images[f"{cam_idx:03d}_{img_idx:04d}"].reshape(
                    self.img_res + [3]) if img_idx > 0 else torch.zeros_like(
                    self.rgb_images[img_key].reshape(self.img_res + [3]))

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def change_idx(self, idx):
        self.idx = idx

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']
