#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
import math
import os
import pickle
import time
import cv2
import torch
import lmdb
import json
import copy
import random
import numpy as np
from torch.utils.data.dataset import Dataset
from utils.camera import PerspectiveCamera
from base_dataset import BaseDataset
from kornia.geometry.conversions import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix

# rgb only
from common.utils.pose_utils import joint3DToImg


class SDFDataset_RGB(BaseDataset):
    def __init__(self, db, cfg, mode='train'):
        super(SDFDataset_RGB, self).__init__(db, cfg, mode)
        self.num_sample_points = cfg.num_sample_points
        self.recon_scale = cfg.recon_scale
        self.clamp = cfg.clamp_dist

        if self.use_lmdb and self.mode == 'train':
            if self.hand_branch:
                self.hand_env = lmdb.open(db.sdf_hand_source + '.lmdb', readonly=True, lock=False, readahead=False,
                                          meminit=False)
                with open(os.path.join(db.sdf_hand_source + '.lmdb', 'meta_info.json'), 'r') as f:
                    self.hand_meta = json.load(f)

            if self.obj_branch:
                self.obj_env = lmdb.open(db.sdf_obj_source + '.lmdb', readonly=True, lock=False, readahead=False,
                                         meminit=False)
                with open(os.path.join(db.sdf_obj_source + '.lmdb', 'meta_info.json'), 'r') as f:
                    self.obj_meta = json.load(f)

    def __getitem__(self, index):
        sample_data = copy.deepcopy(self.db[index])

        sample_key = sample_data['id']
        img_path = sample_data['img_path']
        seg_path = sample_data['seg_path']
        bbox = sample_data['bbox']

        hand_side = torch.from_numpy(sample_data['hand_side'])
        try:
            hand_joints_3d = torch.from_numpy(sample_data['hand_joints_3d'])
        except:
            hand_joints_3d = torch.zeros((21, 3))

        try:
            hand_poses = torch.from_numpy(sample_data['hand_poses'])
        except:
            hand_poses = torch.zeros((48))

        try:
            hand_shapes = torch.from_numpy(sample_data['hand_shapes'])
        except:
            hand_shapes = torch.zeros((10))

        try:
            obj_center_3d = torch.from_numpy(sample_data['obj_center_3d'])
        except:
            obj_center_3d = torch.zeros(3)

        try:
            obj_corners_3d = torch.from_numpy(sample_data['obj_corners_3d'])
        except:
            obj_corners_3d = torch.zeros((8, 3))

        try:
            obj_rest_corners_3d = torch.from_numpy(sample_data['obj_rest_corners_3d'])
        except:
            obj_rest_corners_3d = torch.zeros((8, 3))

        try:
            obj_transform = torch.from_numpy(sample_data['obj_transform'])
        except:
            obj_transform = torch.zeros((4, 4))

        if self.mode == 'train':
            sdf_hand_path = sample_data['sdf_hand_path']
            sdf_obj_path = sample_data['sdf_obj_path']
            sdf_scale = torch.from_numpy(sample_data['sdf_scale'])
            sdf_offset = torch.from_numpy(sample_data['sdf_offset'])

        if self.use_lmdb and self.mode == 'train':
            img = self.load_img_lmdb(self.img_env, sample_key, (3, self.input_image_size[0], self.input_image_size[1]))
        else:
            img = self.load_img(img_path)

        camera = PerspectiveCamera(sample_data['fx'], sample_data['fy'], sample_data['cx'], sample_data['cy'])

        if self.mode == 'train':
            trans, scale, rot, do_flip, color_scale = self.get_aug_config(self.dataset_name)
            rot_aug_mat = torch.from_numpy(np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                                     [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                                     [0, 0, 1]], dtype=np.float32))
        else:
            trans, scale, rot, do_flip, color_scale = [0, 0], 1, 0, False, [1.0, 1.0, 1.0]

        if self.use_lmdb and self.mode == 'train':
            img, _ = self.generate_patch_image(img, [0, 0, self.input_image_size[1], self.input_image_size[0]],
                                               self.input_image_size, do_flip, scale, rot)
        else:
            bbox[0] = bbox[0] + bbox[2] * trans[0]
            bbox[1] = bbox[1] + bbox[3] * trans[1]
            img, _ = self.generate_patch_image(img, bbox, self.input_image_size, do_flip, scale, rot)

        for i in range(3):
            img[:, :, i] = np.clip(img[:, :, i] * color_scale[i], 0, 255)

        camera.update_virtual_camera_after_crop(bbox)
        camera.update_intrinsics_after_resize((bbox[-1], bbox[-2]), self.input_image_size)
        rot_cam_extr = torch.from_numpy(camera.extrinsics[:3, :3].T)
        if self.mode == 'train':
            rot_aug_mat = rot_aug_mat @ rot_cam_extr

        if self.mode == 'train' and self.use_inria_aug and random.random() < 0.5:
            random_idx = np.random.randint(low=1, high=1492, size=1, dtype='l')
            inria_key = str(random_idx[0]).rjust(4, '0')
            if self.use_lmdb:
                seg = self.load_seg_lmdb(self.seg_env, sample_key,
                                         (3, self.input_image_size[0], self.input_image_size[1]))
                bg = self.load_img_lmdb(self.inria_env, inria_key,
                                        (3, self.input_image_size[0], self.input_image_size[1]))
            else:
                seg = self.load_seg(seg_path, sample_data['ycb_id'])
                bg = self.load_img(os.path.join(self.inria_aug_source, inria_key + '.jpg'))

            seg, _ = self.generate_patch_image(seg, bbox, self.input_image_size, do_flip, 1.0, rot)
            seg = np.sum(seg, axis=-1, keepdims=True) > 0
            img = seg * img + (1 - seg) * bg
            img = img.astype(np.uint8)

        img = self.image_transform(img)
        cam_intr = torch.from_numpy(camera.intrinsics)
        cam_extr = torch.from_numpy(camera.extrinsics)

        if self.mode == 'train':
            if self.hand_branch and self.obj_branch:
                num_sample_points = self.num_sample_points // 2
            else:
                num_sample_points = self.num_sample_points

            # get points to train sdf
            if self.hand_branch:
                if self.use_lmdb:
                    hand_samples, hand_labels = self.unpack_sdf_lmdb(self.hand_env, sample_key, self.hand_meta,
                                                                     num_sample_points, hand=True, clamp=self.clamp,
                                                                     filter_dist=True)
                else:
                    hand_samples, hand_labels = self.unpack_sdf(sdf_hand_path, num_sample_points, hand=True,
                                                                clamp=self.clamp, filter_dist=True)
                hand_samples[:, 0:3] = hand_samples[:, 0:3] / sdf_scale - sdf_offset
                hand_labels = hand_labels.long()
                if 'ho3d' in self.dataset_name:
                    hand_samples[:, 1:3] *= -1
            else:
                hand_samples = torch.zeros((num_sample_points, 5), dtype=torch.float32)
                hand_labels = -1. * torch.ones(num_sample_points, dtype=torch.int32)

            if self.obj_branch:
                if self.use_lmdb:
                    obj_samples, obj_labels = self.unpack_sdf_lmdb(self.obj_env, sample_key, self.obj_meta,
                                                                   num_sample_points, hand=False, clamp=self.clamp,
                                                                   filter_dist=True)
                else:
                    obj_samples, obj_labels = self.unpack_sdf(sdf_obj_path, num_sample_points, hand=False,
                                                              clamp=self.clamp, filter_dist=True)
                obj_samples[:, 0:3] = obj_samples[:, 0:3] / sdf_scale - sdf_offset
                obj_lablels = obj_labels.long()
                if 'ho3d' in self.dataset_name:
                    obj_samples[:, 1:3] *= -1
            else:
                obj_samples = torch.zeros((num_sample_points, 5), dtype=torch.float32)
                obj_labels = -1. * torch.ones(num_sample_points, dtype=torch.int32)

            hand_samples[:, 0:3] = torch.mm(rot_aug_mat, hand_samples[:, 0:3].transpose(1, 0)).transpose(1, 0)
            obj_samples[:, 0:3] = torch.mm(rot_aug_mat, obj_samples[:, 0:3].transpose(1, 0)).transpose(1, 0)
            hand_joints_3d[:, 0:3] = torch.mm(rot_aug_mat, hand_joints_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)
            hand_poses[:3] = rotation_matrix_to_angle_axis(
                rot_aug_mat @ angle_axis_to_rotation_matrix(hand_poses[:3].unsqueeze(0))).squeeze(0)
            obj_corners_3d[:, 0:3] = torch.mm(rot_aug_mat, obj_corners_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)
            obj_center_3d = torch.mm(rot_aug_mat, obj_center_3d.unsqueeze(1)).squeeze()
            trans_with_rot = torch.zeros((4, 4))
            trans_with_rot[:3, :3] = rot_aug_mat
            trans_with_rot[3, 3] = 1.
            obj_transform = torch.mm(trans_with_rot, obj_transform)
            obj_transform[:3, 3] = obj_transform[:3, 3] - hand_joints_3d[0]

            hand_center_3d = hand_joints_3d[0]
            hand_samples[:, 0:3] = (hand_samples[:, 0:3] - hand_center_3d) * self.recon_scale
            hand_samples[:, 3:] = hand_samples[:, 3:] / sdf_scale * self.recon_scale
            hand_samples[:, 0:5] = hand_samples[:, 0:5] / 2

            obj_samples[:, 0:3] = (obj_samples[:, 0:3] - hand_center_3d) * self.recon_scale
            obj_samples[:, 3:] = obj_samples[:, 3:] / sdf_scale * self.recon_scale
            obj_samples[:, 0:5] = obj_samples[:, 0:5] / 2

            input_iter = dict(img=img)
            label_iter = dict(hand_sdf=hand_samples, hand_labels=hand_labels, obj_sdf=obj_samples,
                              obj_labels=obj_labels, hand_joints_3d=hand_joints_3d, obj_center_3d=obj_center_3d,
                              obj_corners_3d=obj_corners_3d)
            meta_iter = dict(cam_intr=cam_intr, cam_extr=cam_extr, id=sample_key, hand_center_3d=hand_center_3d,
                             hand_poses=hand_poses, hand_shapes=hand_shapes, obj_rest_corners_3d=obj_rest_corners_3d,
                             obj_transform=obj_transform)

            return input_iter, label_iter, meta_iter
        else:
            hand_center_3d = torch.mm(rot_cam_extr, hand_joints_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)[0]
            trans_with_rot = torch.zeros((4, 4))
            trans_with_rot[:3, :3] = rot_cam_extr
            trans_with_rot[3, 3] = 1.
            obj_transform = torch.mm(trans_with_rot, obj_transform)
            obj_transform[:3, 3] = obj_transform[:3, 3] - hand_center_3d

            input_iter = dict(img=img)
            meta_iter = dict(cam_intr=cam_intr, cam_extr=cam_extr, id=sample_key, hand_center_3d=hand_center_3d,
                             hand_poses=hand_poses, hand_shapes=hand_shapes, obj_rest_corners_3d=obj_rest_corners_3d,
                             obj_transform=obj_transform)

            return input_iter, meta_iter

    def unpack_sdf_lmdb(self, env, key, meta, subsample=None, hand=True, clamp=None, filter_dist=False):
        """
        @description: unpack sdf samples stored in the lmdb dataset.
        ---------
        @param: lmdb env, sample key, lmdb meta, num points, whether is hand, clamp dist, whether filter
        -------
        @Returns: points with sdf, part labels (only meaningful for hands)
        -------
        """

        def filter_invalid_sdf_lmdb(tensor, dist):
            keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
            return tensor[keep, :]

        def remove_nans(tensor):
            tensor_nan = torch.isnan(tensor[:, 3])
            return tensor[~tensor_nan, :]

        with env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))

        index = meta['keys'].index(key)
        npz = np.array(np.frombuffer(buf, dtype=np.float32))
        pos_num = meta['pos_num'][index]
        neg_num = meta['neg_num'][index]
        feat_dim = meta['dim']
        total_num = pos_num + neg_num
        npz = npz.reshape((-1, feat_dim))[:total_num, :]

        try:
            pos_tensor = remove_nans(torch.from_numpy(npz[:pos_num, :]))
            neg_tensor = remove_nans(torch.from_numpy(npz[pos_num:, :]))
        except Exception as e:
            print("fail to load {}, {}".format(key, e))

        # split the sample into half
        half = int(subsample / 2)

        if filter_dist:
            pos_tensor = filter_invalid_sdf_lmdb(pos_tensor, 2.0)
            neg_tensor = filter_invalid_sdf_lmdb(neg_tensor, 2.0)

        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

        samples_and_labels = torch.cat([sample_pos, sample_neg], 0)
        samples = samples_and_labels[:, :-1]
        labels = samples_and_labels[:, -1]

        if clamp:
            labels[samples[:, 3] < -clamp] = -1
            labels[samples[:, 3] > clamp] = -1

        if not hand:
            labels[:] = -1

        return samples, labels

    def unpack_sdf(self, data_path, subsample=None, hand=True, clamp=None, filter_dist=False):
        """
        @description: unpack sdf samples.
        ---------
        @param: sdf data path, num points, whether is hand, clamp dist, whether filter
        -------
        @Returns: points with sdf, part labels (only meaningful for hands)
        -------
        """

        def filter_invalid_sdf(tensor, lab_tensor, dist):
            keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
            return tensor[keep, :], lab_tensor[keep, :]

        def remove_nans(tensor):
            tensor_nan = torch.isnan(tensor[:, 3])
            return tensor[~tensor_nan, :]

        npz = np.load(data_path)

        try:
            pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
            neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
            pos_sdf_other = torch.from_numpy(npz["pos_other"])
            neg_sdf_other = torch.from_numpy(npz["neg_other"])
            if hand:
                lab_pos_tensor = torch.from_numpy(npz["lab_pos"])
                lab_neg_tensor = torch.from_numpy(npz["lab_neg"])
            else:
                lab_pos_tensor = torch.from_numpy(npz["lab_pos_other"])
                lab_neg_tensor = torch.from_numpy(npz["lab_neg_other"])
        except Exception as e:
            print("fail to load {}, {}".format(data_path, e))

        if hand:
            pos_tensor = torch.cat([pos_tensor, pos_sdf_other], 1)
            neg_tensor = torch.cat([neg_tensor, neg_sdf_other], 1)
        else:
            xyz_pos = pos_tensor[:, :3]
            sdf_pos = pos_tensor[:, 3].unsqueeze(1)
            pos_tensor = torch.cat([xyz_pos, pos_sdf_other, sdf_pos], 1)

            xyz_neg = neg_tensor[:, :3]
            sdf_neg = neg_tensor[:, 3].unsqueeze(1)
            neg_tensor = torch.cat([xyz_neg, neg_sdf_other, sdf_neg], 1)

        # split the sample into half
        half = int(subsample / 2)

        if filter_dist:
            pos_tensor, lab_pos_tensor = filter_invalid_sdf(pos_tensor, lab_pos_tensor, 2.0)
            neg_tensor, lab_neg_tensor = filter_invalid_sdf(neg_tensor, lab_neg_tensor, 2.0)

        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

        # label
        sample_pos_lab = torch.index_select(lab_pos_tensor, 0, random_pos)
        sample_neg_lab = torch.index_select(lab_neg_tensor, 0, random_neg)

        # hand part label
        # 0-finger corase, 1-finger fine, 2-contact, 3-sealed wrist
        hand_part_pos = sample_pos_lab[:, 0]
        hand_part_neg = sample_neg_lab[:, 0]
        samples = torch.cat([sample_pos, sample_neg], 0)
        labels = torch.cat([hand_part_pos, hand_part_neg], 0)

        if clamp:
            labels[samples[:, 3] < -clamp] = -1
            labels[samples[:, 3] > clamp] = -1

        if not hand:
            labels[:] = -1

        return samples, labels


# rgb-d image only
class SDFDataset_RGBD(BaseDataset):
    def __init__(self, db, cfg, mode='train',cube = 500):
        super(SDFDataset_RGBD, self).__init__(db, cfg, mode)
        self.num_sample_points = cfg.num_sample_points
        self.recon_scale = cfg.recon_scale
        self.clamp = cfg.clamp_dist
        self.cube_size = cube
        
        if self.use_lmdb and self.mode == 'train':
            if self.hand_branch:
                self.hand_env = lmdb.open(db.sdf_hand_source + '.lmdb', readonly=True, lock=False, readahead=False, meminit=False)
                with open(os.path.join(db.sdf_hand_source + '.lmdb', 'meta_info.json'), 'r') as f:
                   self.hand_meta = json.load(f)

            if self.obj_branch:
                self.obj_env = lmdb.open(db.sdf_obj_source + '.lmdb', readonly=True, lock=False, readahead=False, meminit=False)
                with open(os.path.join(db.sdf_obj_source + '.lmdb', 'meta_info.json'), 'r') as f:
                   self.obj_meta = json.load(f)

    def __getitem__(self, index):
        sample_data = copy.deepcopy(self.db[index])

        sample_key = sample_data['id']
        img_path = sample_data['img_path']
        seg_path = sample_data['seg_path']
        bbox = sample_data['bbox']
        hand_side = torch.from_numpy(sample_data['hand_side'])
        try:
            hand_joints_3d = torch.from_numpy(sample_data['hand_joints_3d'])
            center_xyz = hand_joints_3d.mean(0)
            center_xyz = center_xyz.detach().cpu().numpy()
            center_xyz = center_xyz*1000
        except:
            hand_joints_3d = torch.zeros((21, 3))

        try:
            hand_poses = torch.from_numpy(sample_data['hand_poses'])
        except:
            hand_poses = torch.zeros((48))

        try:
            hand_shapes = torch.from_numpy(sample_data['hand_shapes'])
        except:
            hand_shapes = torch.zeros((10))
        
        try:
            obj_center_3d = torch.from_numpy(sample_data['obj_center_3d'])
        except:
            obj_center_3d = torch.zeros(3)

        try:
            obj_corners_3d = torch.from_numpy(sample_data['obj_corners_3d'])
        except:
            obj_corners_3d = torch.zeros((8, 3))

        try:
            obj_rest_corners_3d = torch.from_numpy(sample_data['obj_rest_corners_3d'])
        except:
            obj_rest_corners_3d = torch.zeros((8, 3))

        try:
            obj_transform = torch.from_numpy(sample_data['obj_transform'])
        except:
            obj_transform = torch.zeros((4, 4))

        if self.mode == 'train':
            sdf_hand_path = sample_data['sdf_hand_path']
            sdf_obj_path = sample_data['sdf_obj_path']
            sdf_scale = torch.from_numpy(sample_data['sdf_scale'])
            sdf_offset = torch.from_numpy(sample_data['sdf_offset'])

        if self.use_lmdb and self.mode == 'train':
            img = self.load_img_lmdb(self.img_env, sample_key, (3, self.input_image_size[0], self.input_image_size[1]))
        else:
            img = self.load_img(img_path)
        img_d = cv2.imread(img_path.replace('color_', 'aligned_depth_to_color_').replace('jpg', 'png'),cv2.IMREAD_ANYDEPTH)
        
        camera = PerspectiveCamera(sample_data['fx'], sample_data['fy'], sample_data['cx'], sample_data['cy'])
        img_d = img_d.astype(np.float32)

        if self.mode == 'train':
            trans, scale, rot, do_flip, color_scale = self.get_aug_config(self.dataset_name)
            rot_aug_mat = torch.from_numpy(np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0], [0, 0, 1]], dtype=np.float32))
        else:
            trans, scale, rot, do_flip, color_scale = [0, 0], 1, 0, False, [1.0, 1.0, 1.0]

        if self.use_lmdb and self.mode == 'train':
            img, _ = self.generate_patch_image(img, [0, 0, self.input_image_size[1], self.input_image_size[0]], self.input_image_size, do_flip, scale, rot)
            b = copy.deepcopy(bbox)
            b[0] = bbox[0] + bbox[2] * trans[0]
            b[1] = bbox[1] + bbox[3] * trans[1]
            img_d, _ = self.generate_patch_image_depth(img_d, b, self.input_image_size,center_xyz, self.cube_size,do_flip, scale, rot)
        else:
            bbox[0] = bbox[0] + bbox[2] * trans[0]
            bbox[1] = bbox[1] + bbox[3] * trans[1]
            img, _ = self.generate_patch_image(img, bbox, self.input_image_size, do_flip, scale, rot)
            # img_d
            img_d, _ = self.generate_patch_image_depth(img_d, bbox, self.input_image_size, center_xyz,self.cube_size,do_flip, scale, rot)


        for i in range(3):
            img[:, :, i] = np.clip(img[:, :, i] * color_scale[i], 0, 255)

        camera.update_virtual_camera_after_crop(bbox)
        camera.update_intrinsics_after_resize((bbox[-1], bbox[-2]), self.input_image_size)
        rot_cam_extr = torch.from_numpy(camera.extrinsics[:3, :3].T)
        if self.mode == 'train':
            rot_aug_mat = rot_aug_mat @ rot_cam_extr

        if self.mode == 'train' and self.use_inria_aug and random.random() < 0.5:
            random_idx = np.random.randint(low=1, high=1492, size=1, dtype='l')
            inria_key = str(random_idx[0]).rjust(4, '0')
            if self.use_lmdb:
                seg = self.load_seg_lmdb(self.seg_env, sample_key, (3, self.input_image_size[0], self.input_image_size[1]))
                bg = self.load_img_lmdb(self.inria_env, inria_key, (3, self.input_image_size[0], self.input_image_size[1]))
            else:
                seg = self.load_seg(seg_path, sample_data['ycb_id'])
                bg = self.load_img(os.path.join(self.inria_aug_source, inria_key + '.jpg'))
            
            seg, _ = self.generate_patch_image(seg, bbox, self.input_image_size, do_flip, 1.0, rot)
            seg = np.sum(seg, axis=-1, keepdims=True) > 0
            img = seg * img + (1 - seg) * bg
            img = img.astype(np.uint8)
            # img_d = seg * img_d + (1 - seg) * bg
            # img_d = img_d.astype(np.uint8)

        img = self.image_transform(img)
        img_d = self.normalize_img(img_d.max(),img_d,com=center_xyz,depth_cube=self.cube_size)
        img_d = torch.from_numpy(img_d).float()
        img_d = img_d.unsqueeze(0)

        cam_intr = torch.from_numpy(camera.intrinsics)
        cam_extr = torch.from_numpy(camera.extrinsics)

        if self.mode == 'train':
            if self.hand_branch and self.obj_branch:
                num_sample_points = self.num_sample_points // 2
            else:
                num_sample_points = self.num_sample_points
        
            # get points to train sdf
            if self.hand_branch:
                if self.use_lmdb:
                    hand_samples, hand_labels = self.unpack_sdf_lmdb(self.hand_env, sample_key, self.hand_meta, num_sample_points, hand=True, clamp=self.clamp, filter_dist=True)
                else:
                    hand_samples, hand_labels = self.unpack_sdf(sdf_hand_path, num_sample_points, hand=True, clamp=self.clamp, filter_dist=True)
                hand_samples[:, 0:3] = hand_samples[:, 0:3] / sdf_scale - sdf_offset
                hand_labels = hand_labels.long()
                if 'ho3d' in self.dataset_name:
                    hand_samples[:, 1:3] *= -1
            else:
                hand_samples = torch.zeros((num_sample_points, 5), dtype=torch.float32)
                hand_labels = -1. * torch.ones(num_sample_points, dtype=torch.int32)

            if self.obj_branch:
                if self.use_lmdb:
                    obj_samples, obj_labels = self.unpack_sdf_lmdb(self.obj_env, sample_key, self.obj_meta, num_sample_points, hand=False, clamp=self.clamp, filter_dist=True)
                else:
                    obj_samples, obj_labels = self.unpack_sdf(sdf_obj_path, num_sample_points, hand=False, clamp=self.clamp, filter_dist=True)
                obj_samples[:, 0:3] = obj_samples[:, 0:3] / sdf_scale - sdf_offset
                obj_lablels = obj_labels.long()
                if 'ho3d' in self.dataset_name:
                    obj_samples[:, 1:3] *= -1
            else:
                obj_samples = torch.zeros((num_sample_points, 5), dtype=torch.float32)
                obj_labels = -1. * torch.ones(num_sample_points, dtype=torch.int32)

            hand_samples[:, 0:3] = torch.mm(rot_aug_mat, hand_samples[:, 0:3].transpose(1, 0)).transpose(1, 0)
            obj_samples[:, 0:3] = torch.mm(rot_aug_mat, obj_samples[:, 0:3].transpose(1, 0)).transpose(1, 0)
            hand_joints_3d[:, 0:3] = torch.mm(rot_aug_mat, hand_joints_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)
            hand_poses[:3] = rotation_matrix_to_angle_axis(rot_aug_mat @ angle_axis_to_rotation_matrix(hand_poses[:3].unsqueeze(0))).squeeze(0)
            obj_corners_3d[:, 0:3] = torch.mm(rot_aug_mat, obj_corners_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)
            obj_center_3d = torch.mm(rot_aug_mat, obj_center_3d.unsqueeze(1)).squeeze()
            trans_with_rot = torch.zeros((4, 4))
            trans_with_rot[:3, :3] = rot_aug_mat
            trans_with_rot[3, 3] = 1.
            obj_transform = torch.mm(trans_with_rot, obj_transform)
            obj_transform[:3, 3] = obj_transform[:3, 3] - hand_joints_3d[0]

            hand_center_3d = hand_joints_3d[0]
            hand_samples[:, 0:3] = (hand_samples[:, 0:3] - hand_center_3d) * self.recon_scale
            hand_samples[:, 3:] = hand_samples[:, 3:] / sdf_scale * self.recon_scale
            hand_samples[:, 0:5] = hand_samples[:, 0:5] / 2

            obj_samples[:, 0:3] = (obj_samples[:, 0:3] - hand_center_3d) * self.recon_scale
            obj_samples[:, 3:] = obj_samples[:, 3:] / sdf_scale * self.recon_scale
            obj_samples[:, 0:5] = obj_samples[:, 0:5] / 2

            input_iter = dict(img=img,img_d=img_d)
            label_iter = dict(hand_sdf=hand_samples, hand_labels=hand_labels, obj_sdf=obj_samples, obj_labels=obj_labels, hand_joints_3d=hand_joints_3d, obj_center_3d=obj_center_3d, obj_corners_3d=obj_corners_3d)
            meta_iter = dict(cam_intr=cam_intr, cam_extr=cam_extr, id=sample_key, hand_center_3d=hand_center_3d, hand_poses=hand_poses, hand_shapes=hand_shapes, obj_rest_corners_3d=obj_rest_corners_3d, obj_transform=obj_transform)

            return input_iter, label_iter, meta_iter
        else:
            hand_center_3d = torch.mm(rot_cam_extr, hand_joints_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)[0]
            trans_with_rot = torch.zeros((4, 4))
            trans_with_rot[:3, :3] = rot_cam_extr
            trans_with_rot[3, 3] = 1.
            obj_transform = torch.mm(trans_with_rot, obj_transform)
            obj_transform[:3, 3] = obj_transform[:3, 3] - hand_center_3d

            input_iter = dict(img=img,img_d=img_d)
            meta_iter = dict(cam_intr=cam_intr, cam_extr=cam_extr, id=sample_key, hand_center_3d=hand_center_3d, hand_poses=hand_poses, hand_shapes=hand_shapes, obj_rest_corners_3d=obj_rest_corners_3d, obj_transform=obj_transform)

            return input_iter, meta_iter

        
    def unpack_sdf_lmdb(self, env, key, meta, subsample=None, hand=True, clamp=None, filter_dist=False):
        """
        @description: unpack sdf samples stored in the lmdb dataset.
        ---------
        @param: lmdb env, sample key, lmdb meta, num points, whether is hand, clamp dist, whether filter
        -------
        @Returns: points with sdf, part labels (only meaningful for hands)
        -------
        """

        def filter_invalid_sdf_lmdb(tensor, dist):
            keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
            return tensor[keep, :]

        def remove_nans(tensor):
            tensor_nan = torch.isnan(tensor[:, 3])
            return tensor[~tensor_nan, :]

        with env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))

        index = meta['keys'].index(key)
        npz = np.array(np.frombuffer(buf, dtype=np.float32))
        pos_num = meta['pos_num'][index]
        neg_num = meta['neg_num'][index]
        feat_dim = meta['dim']
        total_num  = pos_num + neg_num
        npz = npz.reshape((-1, feat_dim))[:total_num, :]

        try:
            pos_tensor = remove_nans(torch.from_numpy(npz[:pos_num, :]))
            neg_tensor = remove_nans(torch.from_numpy(npz[pos_num:, :]))
        except Exception as e:
            print("fail to load {}, {}".format(key, e))

        # split the sample into half
        half = int(subsample / 2)

        if filter_dist:
            pos_tensor = filter_invalid_sdf_lmdb(pos_tensor, 2.0)
            neg_tensor = filter_invalid_sdf_lmdb(neg_tensor, 2.0)

        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

        samples_and_labels = torch.cat([sample_pos, sample_neg], 0)
        samples = samples_and_labels[:, :-1]
        labels = samples_and_labels[:, -1]

        if clamp:
            labels[samples[:, 3] < -clamp] = -1
            labels[samples[:, 3] > clamp] = -1

        if not hand:
            labels[:] = -1

        return samples, labels

    def unpack_sdf(self, data_path, subsample=None, hand=True, clamp=None, filter_dist=False):
        """
        @description: unpack sdf samples.
        ---------
        @param: sdf data path, num points, whether is hand, clamp dist, whether filter
        -------
        @Returns: points with sdf, part labels (only meaningful for hands)
        -------
        """

        def filter_invalid_sdf(tensor, lab_tensor, dist):
            keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
            return tensor[keep, :], lab_tensor[keep, :]

        def remove_nans(tensor):
            tensor_nan = torch.isnan(tensor[:, 3])
            return tensor[~tensor_nan, :]

        npz = np.load(data_path)

        try:
            pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
            neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
            pos_sdf_other = torch.from_numpy(npz["pos_other"])
            neg_sdf_other = torch.from_numpy(npz["neg_other"])
            if hand:
                lab_pos_tensor = torch.from_numpy(npz["lab_pos"])
                lab_neg_tensor = torch.from_numpy(npz["lab_neg"])
            else:
                lab_pos_tensor = torch.from_numpy(npz["lab_pos_other"])
                lab_neg_tensor = torch.from_numpy(npz["lab_neg_other"])
        except Exception as e:
            print("fail to load {}, {}".format(data_path, e))

        if hand:
            pos_tensor = torch.cat([pos_tensor, pos_sdf_other], 1)
            neg_tensor = torch.cat([neg_tensor, neg_sdf_other], 1)
        else:
            xyz_pos = pos_tensor[:, :3]
            sdf_pos = pos_tensor[:, 3].unsqueeze(1)
            pos_tensor = torch.cat([xyz_pos, pos_sdf_other, sdf_pos], 1)

            xyz_neg = neg_tensor[:, :3]
            sdf_neg = neg_tensor[:, 3].unsqueeze(1)
            neg_tensor = torch.cat([xyz_neg, neg_sdf_other, sdf_neg], 1)

        # split the sample into half
        half = int(subsample / 2)

        if filter_dist:
            pos_tensor, lab_pos_tensor = filter_invalid_sdf(pos_tensor, lab_pos_tensor, 2.0)
            neg_tensor, lab_neg_tensor = filter_invalid_sdf(neg_tensor, lab_neg_tensor, 2.0)

        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

        # label
        sample_pos_lab = torch.index_select(lab_pos_tensor, 0, random_pos)
        sample_neg_lab = torch.index_select(lab_neg_tensor, 0, random_neg)

        # hand part label
        # 0-finger corase, 1-finger fine, 2-contact, 3-sealed wrist
        hand_part_pos = sample_pos_lab[:, 0]
        hand_part_neg = sample_neg_lab[:, 0]
        samples = torch.cat([sample_pos, sample_neg], 0)
        labels = torch.cat([hand_part_pos, hand_part_neg], 0)

        if clamp:
            labels[samples[:, 3] < -clamp] = -1
            labels[samples[:, 3] > clamp] = -1

        if not hand:
            labels[:] = -1

        return samples, labels


# RGB+D+PCL
class SDFDataset(BaseDataset):
    def __init__(self, db, cfg, mode='train', cube=500):
        super(SDFDataset, self).__init__(db, cfg, mode)
        self.num_sample_points = cfg.num_sample_points
        self.recon_scale = cfg.recon_scale
        self.clamp = cfg.clamp_dist
        self.cube_size = cube
        self.pcl_sample_nums = cfg.pcl_sample_nums
        self.heatmap = cfg.heatmap
        self.add_info = cfg.add_info
        #self.dataset_name = cfg.trainset_3d

        if self.use_lmdb and self.mode == 'train':
            if self.hand_branch:
                self.hand_env = lmdb.open(db.sdf_hand_source + '.lmdb', readonly=True, lock=False, readahead=False,
                                          meminit=False)
                with open(os.path.join(db.sdf_hand_source + '.lmdb', 'meta_info.json'), 'r') as f:
                    self.hand_meta = json.load(f)

            if self.obj_branch:
                self.obj_env = lmdb.open(db.sdf_obj_source + '.lmdb', readonly=True, lock=False, readahead=False,
                                         meminit=False)
                with open(os.path.join(db.sdf_obj_source + '.lmdb', 'meta_info.json'), 'r') as f:
                    self.obj_meta = json.load(f)

    def __getitem__(self, index):
        sample_data = copy.deepcopy(self.db[index])

        sample_key = sample_data['id']

        img_path = sample_data['img_path']
        seg_path = sample_data['seg_path']
        bbox = sample_data['bbox']

        hand_side = torch.from_numpy(sample_data['hand_side'])
        try:
            hand_joints_3d = torch.from_numpy(sample_data['hand_joints_3d'])
            center_xyz = hand_joints_3d.mean(0)
            center_xyz = center_xyz.detach().cpu().numpy()
            center_xyz = center_xyz * 1000
        except:
            hand_joints_3d = torch.zeros((21, 3))

        try:
            hand_poses = torch.from_numpy(sample_data['hand_poses'])
        except:
            hand_poses = torch.zeros((48))

        try:
            hand_shapes = torch.from_numpy(sample_data['hand_shapes'])
        except:
            hand_shapes = torch.zeros((10))

        try:
            obj_center_3d = torch.from_numpy(sample_data['obj_center_3d'])
        except:
            obj_center_3d = torch.zeros(3)

        try:
            obj_corners_3d = torch.from_numpy(sample_data['obj_corners_3d'])
        except:
            obj_corners_3d = torch.zeros((8, 3))

        try:
            obj_rest_corners_3d = torch.from_numpy(sample_data['obj_rest_corners_3d'])
        except:
            obj_rest_corners_3d = torch.zeros((8, 3))

        try:
            obj_transform = torch.from_numpy(sample_data['obj_transform'])
        except:
            obj_transform = torch.zeros((4, 4))

        if self.mode == 'train':
            sdf_hand_path = sample_data['sdf_hand_path']
            sdf_obj_path = sample_data['sdf_obj_path']
            sdf_scale = torch.from_numpy(sample_data['sdf_scale'])
            sdf_offset = torch.from_numpy(sample_data['sdf_offset'])

            # load hm
            if self.heatmap:
                if 'dexycb' in self.dataset_name:
                    subject = int(img_path.split('/')[-4].split('-')[-1])
                    video_id = img_path.split('/')[-3]
                    sub_video_id = img_path.split('/')[-2]
                    frame_idx = int(img_path.split('/')[-1].split('_')[-1].split('.')[0])

                    hm_file_name = '_'.join([str(subject), video_id, sub_video_id, str(frame_idx)])


                    hm_h = np.load('/home/cyc/pycharm/data/hand/DexYCB/heatmap/heatmap_hand_0.2_3_10/'+hm_file_name+'.npy')
                    hm_o = np.load('/home/cyc/pycharm/data/hand/DexYCB/heatmap/heatmap_obj_0.2_3_10/' + hm_file_name+'.npy')
                    hm_ho = torch.from_numpy(np.stack([hm_h,hm_o],axis=0))
                elif 'obman' in self.dataset_name:

                    hm_h = np.load('/home/cyc/pycharm/data/hand/obman/train/heatmap/heatmap_hand_0.2_3_5/' + sample_key + '.npy')
                    hm_o = np.load('/home/cyc/pycharm/data/hand/obman/train/heatmap/heatmap_obj_0.2_3_5/' + sample_key + '.npy')
                    hm_ho = torch.from_numpy(np.stack([hm_h, hm_o], axis=0))

            else:
                hm_ho = torch.zeros(0)
        else:
            hm_ho = torch.zeros(0)
            # # load hm
            # if self.heatmap:
            #     subject = int(img_path.split('/')[-4].split('-')[-1])
            #     video_id = img_path.split('/')[-3]
            #     sub_video_id = img_path.split('/')[-2]
            #     frame_idx = int(img_path.split('/')[-1].split('_')[-1].split('.')[0])
            #
            #     hm_file_name = '_'.join([str(subject), video_id, sub_video_id, str(frame_idx)])
            #     hm_h = np.load('/home/cyc/pycharm/data/hand/DexYCB/heatmap_hand_gs_test_xy/' + hm_file_name + '.npy')
            #     hm_o = np.load('/home/cyc/pycharm/data/hand/DexYCB/heatmap_obj_gs_test_xy/' + hm_file_name + '.npy')
            #     hm_ho = torch.from_numpy(np.stack([hm_h, hm_o], axis=0))
            # else:
            #     hm_ho = torch.zeros(0)


        if self.use_lmdb and self.mode == 'train':
            img = self.load_img_lmdb(self.img_env, sample_key, (3, self.input_image_size[0], self.input_image_size[1]))
        else:
            img = self.load_img(img_path)
        if 'obman' in self.dataset_name:
            img_d = cv2.imread(img_path.replace('rgb', 'depth').replace('jpg', 'png'),1)
            data_root = '../datasets/obman'
            meta_data_file = os.path.join(data_root,'data', self.mode,'meta', sample_key+'.pkl')
            with open(meta_data_file, 'rb') as f:
                meta_info = pickle.load(f)

            img_d = img_d[:, :, 0]
            depth_max = meta_info['depth_max']
            depth_min = meta_info['depth_min']
            assert img_d.max() == 255, 'Max value of depth jpg should be 255, not {}'.format(img_d.max())
            img_d = ((img_d - 1) / 254 * (depth_min - depth_max) + depth_max)*1000

        else:
            img_d = cv2.imread(img_path.replace('color_', 'aligned_depth_to_color_').replace('jpg', 'png'),cv2.IMREAD_ANYDEPTH)

        camera = PerspectiveCamera(sample_data['fx'], sample_data['fy'], sample_data['cx'], sample_data['cy'])
        img_d = img_d.astype(np.float32)

        if self.mode == 'train':
            trans, scale, rot, do_flip, color_scale = self.get_aug_config(self.dataset_name)
            rot_aug_mat = torch.from_numpy(np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                                     [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                                     [0, 0, 1]], dtype=np.float32))
        else:
            trans, scale, rot, do_flip, color_scale = [0, 0], 1, 0, False, [1.0, 1.0, 1.0]

        if self.use_lmdb and self.mode == 'train':
            img, _ = self.generate_patch_image(img, [0, 0, self.input_image_size[1], self.input_image_size[0]],
                                               self.input_image_size, do_flip, scale, rot)
            b = copy.deepcopy(bbox)
            b[0] = bbox[0] + bbox[2] * trans[0]
            b[1] = bbox[1] + bbox[3] * trans[1]
            img_d, _ = self.generate_patch_image_depth(img_d, b, self.input_image_size, center_xyz, self.cube_size,
                                                       do_flip, scale, rot)
        else:
            bbox[0] = bbox[0] + bbox[2] * trans[0]
            bbox[1] = bbox[1] + bbox[3] * trans[1]
            img, _ = self.generate_patch_image(img, bbox, self.input_image_size, do_flip, scale, rot)
            # img_d
            img_d, _ = self.generate_patch_image_depth(img_d, bbox, self.input_image_size, center_xyz, self.cube_size,
                                                       do_flip, scale, rot)

        for i in range(3):
            img[:, :, i] = np.clip(img[:, :, i] * color_scale[i], 0, 255)
        # 更新相机内参
        camera.update_virtual_camera_after_crop(bbox)
        camera.update_intrinsics_after_resize((bbox[-1], bbox[-2]), self.input_image_size)
        rot_cam_extr = torch.from_numpy(camera.extrinsics[:3, :3].T)
        if self.mode == 'train':
            #M = copy.deepcopy(rot_aug_mat)
            rot_aug_mat = rot_aug_mat @ rot_cam_extr
            center_xyz =torch.mm(rot_aug_mat, torch.from_numpy(center_xyz).unsqueeze(1)).squeeze()
        else:
            center_xyz = torch.mm(rot_cam_extr, torch.from_numpy(center_xyz).unsqueeze(1)).squeeze()

        center_xyz = np.array(center_xyz)

        if self.mode == 'train' and self.use_inria_aug and random.random() < 0.5:
            random_idx = np.random.randint(low=1, high=1492, size=1, dtype='l')
            inria_key = str(random_idx[0]).rjust(4, '0')
            if self.use_lmdb:
                seg = self.load_seg_lmdb(self.seg_env, sample_key,
                                         (3, self.input_image_size[0], self.input_image_size[1]))
                bg = self.load_img_lmdb(self.inria_env, inria_key,
                                        (3, self.input_image_size[0], self.input_image_size[1]))
            else:
                seg = self.load_seg(seg_path, sample_data['ycb_id'])
                bg = self.load_img(os.path.join(self.inria_aug_source, inria_key + '.jpg'))

            seg, _ = self.generate_patch_image(seg, bbox, self.input_image_size, do_flip, 1.0, rot)
            seg = np.sum(seg, axis=-1, keepdims=True) > 0
            img = seg * img + (1 - seg) * bg
            img = img.astype(np.uint8)
            # img_d = seg * img_d + (1 - seg) * bg
            # img_d = img_d.astype(np.uint8)

        img = self.image_transform(img)
        img_d = self.normalize_img(img_d.max(), img_d, com=center_xyz, depth_cube=self.cube_size)

        # get pcl
        cube_x,cube_y = self.bbox2cube(bbox,center_xyz[2],camera.intrinsics)
        cube = [cube_x,cube_y, self.cube_size]
        #cube = [bbox[2], bbox[3], self.cube_size]
        cube = np.array(cube)

        pcl = self.getpcl(img_d, center_xyz, cube, camera.intrinsics)
        pcl_index = np.arange(pcl.shape[0])
        pcl_num = pcl.shape[0]
        if pcl_num == 0:
            pcl_sample = np.zeros([self.pcl_sample_nums, 3])
        else:
            if pcl_num < self.pcl_sample_nums:
                tmp = math.floor(self.pcl_sample_nums / pcl_num)
                index_temp = pcl_index.repeat(tmp)
                pcl_index = np.append(index_temp, np.random.choice(pcl_index, size=divmod(self.pcl_sample_nums, pcl_num)[1],
                                                                   replace=False))
            select = np.random.choice(pcl_index, self.pcl_sample_nums, replace=False)
            pcl_sample = pcl[select, :]


        cam_intr = torch.from_numpy(camera.intrinsics)
        cam_extr = torch.from_numpy(camera.extrinsics)
        pcl_sample = torch.from_numpy(pcl_sample).float()
        cube = torch.from_numpy(cube).float()
        img_d = torch.from_numpy(img_d).float()
        img_d = img_d.unsqueeze(0)

        # VISUAL PCL
        if index%5000==0 :
            np.savetxt('/home/cyc/pycharm/lxy/visual//pcl_%d.txt' % (index), pcl_sample, fmt='%.3f')

            img_show = cv2.cvtColor(img_d.cpu().numpy()[0]*255, cv2.COLOR_GRAY2RGB)

            cv2.imwrite('/home/cyc/pycharm/lxy/visual//pcl_%d.png' % (index), img_show)

        if self.mode == 'train':
            if self.hand_branch and self.obj_branch:
                num_sample_points = self.num_sample_points // 2
            else:#
                num_sample_points = self.num_sample_points

            # get points to train sdf
            if self.hand_branch:
                if self.use_lmdb:
                    hand_samples, hand_labels = self.unpack_sdf_lmdb(self.hand_env, sample_key, self.hand_meta,
                                                                     num_sample_points, hand=True, clamp=self.clamp,
                                                                     filter_dist=True)
                else:
                    hand_samples, hand_labels = self.unpack_sdf(sdf_hand_path, num_sample_points, hand=True,
                                                                clamp=self.clamp, filter_dist=True)
                hand_samples[:, 0:3] = hand_samples[:, 0:3] / sdf_scale - sdf_offset
                hand_labels = hand_labels.long()
                if 'ho3d' in self.dataset_name:
                    hand_samples[:, 1:3] *= -1
            else:
                hand_samples = torch.zeros((num_sample_points, 5), dtype=torch.float32)
                hand_labels = -1. * torch.ones(num_sample_points, dtype=torch.int32)

            if self.obj_branch:
                if self.use_lmdb:
                    obj_samples, obj_labels = self.unpack_sdf_lmdb(self.obj_env, sample_key, self.obj_meta,
                                                                   num_sample_points, hand=False, clamp=self.clamp,
                                                                   filter_dist=True)
                else:
                    obj_samples, obj_labels = self.unpack_sdf(sdf_obj_path, num_sample_points, hand=False,
                                                              clamp=self.clamp, filter_dist=True)
                obj_samples[:, 0:3] = obj_samples[:, 0:3] / sdf_scale - sdf_offset
                obj_lablels = obj_labels.long()
                if 'ho3d' in self.dataset_name:
                    obj_samples[:, 1:3] *= -1
            else:
                obj_samples = torch.zeros((num_sample_points, 5), dtype=torch.float32)
                obj_labels = -1. * torch.ones(num_sample_points, dtype=torch.int32)

            hand_samples[:, 0:3] = torch.mm(rot_aug_mat, hand_samples[:, 0:3].transpose(1, 0)).transpose(1, 0)
            obj_samples[:, 0:3] = torch.mm(rot_aug_mat, obj_samples[:, 0:3].transpose(1, 0)).transpose(1, 0)
            hand_joints_3d[:, 0:3] = torch.mm(rot_aug_mat, hand_joints_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)
            hand_poses[:3] = rotation_matrix_to_angle_axis(
                rot_aug_mat @ angle_axis_to_rotation_matrix(hand_poses[:3].unsqueeze(0))).squeeze(0)
            obj_corners_3d[:, 0:3] = torch.mm(rot_aug_mat, obj_corners_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)
            obj_center_3d = torch.mm(rot_aug_mat, obj_center_3d.unsqueeze(1)).squeeze()
            trans_with_rot = torch.zeros((4, 4))
            trans_with_rot[:3, :3] = rot_aug_mat
            trans_with_rot[3, 3] = 1.
            obj_transform = torch.mm(trans_with_rot, obj_transform)
            obj_transform[:3, 3] = obj_transform[:3, 3] - hand_joints_3d[0]

            hand_center_3d = hand_joints_3d[0]
            hand_samples[:, 0:3] = (hand_samples[:, 0:3] - hand_center_3d) * self.recon_scale
            hand_samples[:, 3:] = hand_samples[:, 3:] / sdf_scale * self.recon_scale
            hand_samples[:, 0:5] = hand_samples[:, 0:5] / 2

            obj_samples[:, 0:3] = (obj_samples[:, 0:3] - hand_center_3d) * self.recon_scale
            obj_samples[:, 3:] = obj_samples[:, 3:] / sdf_scale * self.recon_scale
            obj_samples[:, 0:5] = obj_samples[:, 0:5] / 2

            joint_img = joint3DToImg(hand_joints_3d * 1000, cam_intr.unsqueeze(0))
            joint_img[:,  0:2] = joint_img[:,  0:2] / (self.input_image_size[0] / 2) - 1
            joint_img[:,  2] = (joint_img[:, 2] - center_xyz[2]) / (cube[2] / 2.0)


            input_iter = dict(img=img, img_d=img_d,pcl = pcl_sample)
            label_iter = dict(hand_sdf=hand_samples, hand_labels=hand_labels, obj_sdf=obj_samples,
                              obj_labels=obj_labels, hand_joints_3d=hand_joints_3d, obj_center_3d=obj_center_3d,
                              obj_corners_3d=obj_corners_3d,hm_ho=hm_ho)
            meta_iter = dict(cam_intr=cam_intr, cam_extr=cam_extr, id=sample_key, hand_center_3d=hand_center_3d,
                             hand_poses=hand_poses, hand_shapes=hand_shapes, obj_rest_corners_3d=obj_rest_corners_3d,
                             obj_transform=obj_transform,cube=cube,center_xyz=center_xyz,joint_img=joint_img)

            return input_iter, label_iter, meta_iter
        else:
            hand_center_3d = torch.mm(rot_cam_extr, hand_joints_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)[0]
            trans_with_rot = torch.zeros((4, 4))
            trans_with_rot[:3, :3] = rot_cam_extr
            trans_with_rot[3, 3] = 1.
            obj_transform = torch.mm(trans_with_rot, obj_transform)
            obj_transform[:3, 3] = obj_transform[:3, 3] - hand_center_3d

            input_iter = dict(img=img, img_d=img_d,pcl = pcl_sample)
            meta_iter = dict(cam_intr=cam_intr, cam_extr=cam_extr, id=sample_key, hand_center_3d=hand_center_3d,
                             hand_poses=hand_poses, hand_shapes=hand_shapes, obj_rest_corners_3d=obj_rest_corners_3d,
                             obj_transform=obj_transform,cube=cube,center_xyz=center_xyz,hm_ho=hm_ho)

            return input_iter, meta_iter

    def getpcl(self, imgD, com3D, cube, cam_para=None,M=None):
        mask = np.isclose(imgD, 1)
        dpt_ori = imgD * cube[2] / 2.0 + com3D[2]
        # change the background value
        dpt_ori[mask] = 0

        pcl = (self.depthToPCL(dpt_ori,cam_para) - com3D)
        pcl_num = pcl.shape[0]
        cube_tile = np.tile(cube / 2.0, pcl_num).reshape([pcl_num, 3])
        pcl = pcl / cube_tile
        return pcl

    def depthToPCL(self, dpt, paras, T=None,background_val=0.):

        # fx, fy = paras[0, 0].unsqueeze(1), paras[1, 1].unsqueeze(1)
        # fu, fv = paras[0, 2].unsqueeze(1), paras[1, 2].unsqueeze(1)
        fx, fy = paras[0, 0], paras[1, 1]
        fu, fv = paras[0, 2], paras[1, 2]
        # get valid points and transform
        pts = np.asarray(np.where(~np.isclose(dpt, background_val))).transpose()
        pts = np.concatenate([pts[:, [1, 0]] + 0.5, np.ones((pts.shape[0], 1), dtype='float32')], axis=1)
        if T is not None:
            pts = np.dot(np.linalg.inv(np.asarray(T)), pts.T).T
        pts = (pts[:, 0:2] / pts[:, 2][:, None]).reshape((pts.shape[0], 2))

        # replace the invalid data
        depth = dpt[(~np.isclose(dpt, background_val))]

        # get x and y data in a vectorized way
        row = (pts[:, 0] - fu) / fx * depth
        col = (pts[:, 1] - fv) / fy * depth

        # combine x,y,depth
        return np.column_stack((row, col, depth))

    def unpack_sdf_lmdb(self, env, key, meta, subsample=None, hand=True, clamp=None, filter_dist=False):
        """
        @description: unpack sdf samples stored in the lmdb dataset.
        ---------
        @param: lmdb env, sample key, lmdb meta, num points, whether is hand, clamp dist, whether filter
        -------
        @Returns: points with sdf, part labels (only meaningful for hands)
        -------
        """

        def filter_invalid_sdf_lmdb(tensor, dist):
            keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
            return tensor[keep, :]

        def remove_nans(tensor):
            tensor_nan = torch.isnan(tensor[:, 3])
            return tensor[~tensor_nan, :]

        with env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))

        index = meta['keys'].index(key)
        npz = np.array(np.frombuffer(buf, dtype=np.float32))
        pos_num = meta['pos_num'][index]
        neg_num = meta['neg_num'][index]
        feat_dim = meta['dim']
        total_num = pos_num + neg_num
        npz = npz.reshape((-1, feat_dim))[:total_num, :]

        try:
            pos_tensor = remove_nans(torch.from_numpy(npz[:pos_num, :]))
            neg_tensor = remove_nans(torch.from_numpy(npz[pos_num:, :]))
        except Exception as e:
            print("fail to load {}, {}".format(key, e))

        # split the sample into half
        half = int(subsample / 2)

        if filter_dist:
            pos_tensor = filter_invalid_sdf_lmdb(pos_tensor, 2.0)
            neg_tensor = filter_invalid_sdf_lmdb(neg_tensor, 2.0)

        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

        samples_and_labels = torch.cat([sample_pos, sample_neg], 0)
        samples = samples_and_labels[:, :-1]
        labels = samples_and_labels[:, -1]

        if clamp:
            labels[samples[:, 3] < -clamp] = -1
            labels[samples[:, 3] > clamp] = -1

        if not hand:
            labels[:] = -1

        return samples, labels

    def unpack_sdf(self, data_path, subsample=None, hand=True, clamp=None, filter_dist=False):
        """
        @description: unpack sdf samples.
        ---------
        @param: sdf data path, num points, whether is hand, clamp dist, whether filter
        -------
        @Returns: points with sdf, part labels (only meaningful for hands)
        -------
        """

        def filter_invalid_sdf(tensor, lab_tensor, dist):
            keep = (torch.abs(tensor[:, 3]) < abs(dist)) & (torch.abs(tensor[:, 4]) < abs(dist))
            return tensor[keep, :], lab_tensor[keep, :]

        def remove_nans(tensor):
            tensor_nan = torch.isnan(tensor[:, 3])
            return tensor[~tensor_nan, :]

        npz = np.load(data_path)

        try:
            pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
            neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
            pos_sdf_other = torch.from_numpy(npz["pos_other"])
            neg_sdf_other = torch.from_numpy(npz["neg_other"])
            if hand:
                lab_pos_tensor = torch.from_numpy(npz["lab_pos"])
                lab_neg_tensor = torch.from_numpy(npz["lab_neg"])
            else:
                lab_pos_tensor = torch.from_numpy(npz["lab_pos_other"])
                lab_neg_tensor = torch.from_numpy(npz["lab_neg_other"])
        except Exception as e:
            print("fail to load {}, {}".format(data_path, e))

        if hand:
            pos_tensor = torch.cat([pos_tensor, pos_sdf_other], 1)
            neg_tensor = torch.cat([neg_tensor, neg_sdf_other], 1)
        else:
            xyz_pos = pos_tensor[:, :3]
            sdf_pos = pos_tensor[:, 3].unsqueeze(1)
            pos_tensor = torch.cat([xyz_pos, pos_sdf_other, sdf_pos], 1)

            xyz_neg = neg_tensor[:, :3]
            sdf_neg = neg_tensor[:, 3].unsqueeze(1)
            neg_tensor = torch.cat([xyz_neg, neg_sdf_other, sdf_neg], 1)

        # split the sample into half
        half = int(subsample / 2)

        if filter_dist:
            pos_tensor, lab_pos_tensor = filter_invalid_sdf(pos_tensor, lab_pos_tensor, 2.0)
            neg_tensor, lab_neg_tensor = filter_invalid_sdf(neg_tensor, lab_neg_tensor, 2.0)

        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)

        # label
        sample_pos_lab = torch.index_select(lab_pos_tensor, 0, random_pos)
        sample_neg_lab = torch.index_select(lab_neg_tensor, 0, random_neg)

        # hand part label
        # 0-finger corase, 1-finger fine, 2-contact, 3-sealed wrist
        hand_part_pos = sample_pos_lab[:, 0]
        hand_part_neg = sample_neg_lab[:, 0]
        samples = torch.cat([sample_pos, sample_neg], 0)
        labels = torch.cat([hand_part_pos, hand_part_neg], 0)

        if clamp:
            labels[samples[:, 3] < -clamp] = -1
            labels[samples[:, 3] > clamp] = -1

        if not hand:
            labels[:] = -1

        return samples, labels
        

if __name__ == "__main__":
    from obman.obman import obman
    obman_db = obman('train_30k')
