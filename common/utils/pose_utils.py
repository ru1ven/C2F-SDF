#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import numpy as np
import torch
from torch.nn import functional as F


def soft_argmax(cfg, heatmaps, num_joints):
    depth_dim = heatmaps.shape[1] // num_joints
    H_heatmaps = heatmaps.shape[2]
    W_heatmaps = heatmaps.shape[3]
    heatmaps = heatmaps.reshape((-1, num_joints, depth_dim * H_heatmaps * W_heatmaps))
    heatmaps = F.softmax(heatmaps, 2)
    confidence, _ = torch.max(heatmaps, 2)
    heatmaps = heatmaps.reshape((-1, num_joints, depth_dim, H_heatmaps, W_heatmaps))

    accu_x = heatmaps.sum(dim=(2, 3))
    accu_y = heatmaps.sum(dim=(2, 4))
    accu_z = heatmaps.sum(dim=(3, 4))

    accu_x = accu_x * torch.arange(cfg.heatmap_size[1]).float().cuda()[None, None, :]
    accu_y = accu_y * torch.arange(cfg.heatmap_size[0]).float().cuda()[None, None, :]
    accu_z = accu_z * torch.arange(cfg.heatmap_size[2]).float().cuda()[None, None, :]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    coord_out = torch.cat((accu_x, accu_y, accu_z), dim=2)

    return coord_out, confidence


def decode_volume(cfg, heatmaps, center3d, cam_intr):
    hm_pred = heatmaps.clone()
    hm_pred[:, :, 0] *= (cfg.image_size[1] // cfg.heatmap_size[1])
    hm_pred[:, :, 1] *= (cfg.image_size[0] // cfg.heatmap_size[0])
    hm_pred[:, :, 2] = (hm_pred[:, :, 2] / cfg.heatmap_size[2] * 2 - 1) * cfg.depth_dim + center3d[:, [2]]

    fx = cam_intr[:, 0, 0].unsqueeze(1)
    fy = cam_intr[:, 1, 1].unsqueeze(1)
    cx = cam_intr[:, 0, 2].unsqueeze(1)
    cy = cam_intr[:, 1, 2].unsqueeze(1)

    cam_x = ((hm_pred[:, :, 0] - cx) / fx * hm_pred[:, :, 2]).unsqueeze(2)
    cam_y = ((hm_pred[:, :, 1] - cy) / fy * hm_pred[:, :, 2]).unsqueeze(2)
    cam_z = hm_pred[:, :, [2]]
    cam_coords = torch.cat([cam_x, cam_y, cam_z], 2)

    return cam_coords


def decode_volume_abs(cfg, heatmaps, cam_intr):
    # please refer to the paper "Hand Pose Estimation via Latent 2.5D Heatmap Regression" for more details.
    norm_coords = heatmaps.clone()
    norm_coords[:, :, 0] *= (cfg.image_size[1] // cfg.heatmap_size[1])
    norm_coords[:, :, 1] *= (cfg.image_size[0] // cfg.heatmap_size[0])
    norm_coords[:, :, 2] = (norm_coords[:, :, 2] / cfg.heatmap_size[2] * 2 - 1) * cfg.depth_dim

    fx, fy = cam_intr[:, 0, 0], cam_intr[:, 1, 1]
    cx, cy = cam_intr[:, 0, 2], cam_intr[:, 1, 2]

    x_n, x_m = (norm_coords[:, 3, 0] - cx) / fx, (norm_coords[:, 2, 0] - cx) / fx
    y_n, y_m = (norm_coords[:, 3, 1] - cy) / fy, (norm_coords[:, 2, 1] - cy) / fy
    z_n, z_m = norm_coords[:, 3, 2], norm_coords[:, 2, 2]

    a = (x_n - x_m) ** 2 + (y_n - y_m) ** 2
    b = 2 * (x_n - x_m) * (x_n * z_n - x_m * z_m) + 2 * (y_n - y_m) * (y_n * z_n - y_m * z_m)
    c = (x_n * z_n - x_m * z_m) ** 2 + (y_n * z_n - y_m * z_m) ** 2 + (z_n - z_m) ** 2 - cfg.norm_factor ** 2

    z_root = 0.5 * (-b + torch.sqrt(b ** 2 - 4 * a * c)) / (a + 1e-7)

    norm_coords[:, :, 2] += z_root.unsqueeze(1)
    cam_x = ((norm_coords[:, :, 0] - cx.unsqueeze(1)) / fx.unsqueeze(1) * norm_coords[:, :, 2]).unsqueeze(2)
    cam_y = ((norm_coords[:, :, 1] - cy.unsqueeze(1)) / fy.unsqueeze(1) * norm_coords[:, :, 2]).unsqueeze(2)
    cam_z = norm_coords[:, :, [2]]
    cam_coords = torch.cat([cam_x, cam_y, cam_z], 2)

    bone_pairs = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    pred_bones = torch.zeros((cam_coords.shape[0], 20)).to(cam_coords.device)
    for index, pair in enumerate(bone_pairs):
        pred_bones[:, index] = torch.norm(cam_coords[:, pair[0]] - cam_coords[:, pair[1]])
    
    if 'obman' in cfg.trainset_3d:
        bone_mean = torch.Tensor([0.04926945, 0.02837802, 0.02505871, 0.03195906, 0.0977657, 0.03123845, 0.02152403, 0.02244521, 0.10214221, 0.02953061, 0.02272312, 0.02512852, 0.09391599, 0.02677647, 0.02259798, 0.02372275, 0.08817818, 0.01826516, 0.01797429, 0.01902172]).to(cam_coords.device)
    else:
        bone_mean = torch.Tensor([0.03919412, 0.03161546, 0.02788814, 0.03607267, 0.0928468, 0.03420997, 0.02304366, 0.02415902, 0.09689835, 0.03286654, 0.02411255, 0.02707138, 0.08777174, 0.0301717, 0.02593414, 0.02469868, 0.08324047, 0.02167141, 0.0196476 , 0.02105321]).to(cam_coords.device)
    
    optim_scale = torch.sum(pred_bones * bone_mean, 1) / (torch.sum(pred_bones ** 2, 1) + 1e-7)
    cam_coords = cam_coords * optim_scale

    return cam_coords


def joint3DToImg(xyz, cam_intr=None,flip=1 ):
    fx, fy = cam_intr[:, 0, 0].unsqueeze(1), cam_intr[:, 1, 1].unsqueeze(1)
    fu, fv = cam_intr[:, 0, 2].unsqueeze(1), cam_intr[:, 1, 2].unsqueeze(1)

    ret = torch.zeros_like(xyz,dtype=torch.float32)

    if len(ret.shape) == 1:
        ret[0] = (xyz[0] * fx / xyz[2] + fu)
        ret[1] = (flip * xyz[1] * fy / xyz[2] + fv)
        ret[2] = xyz[2]
    elif len(ret.shape) == 2:
        ret[:, 0] = (xyz[:, 0] * fx / xyz[:, 2] + fu)
        ret[:, 1] = (flip * xyz[:, 1] * fy / xyz[:, 2] + fv)
        ret[:, 2] = xyz[:, 2]
    else:
        ret[:, :, 0] = (xyz[:, :, 0] * fx / xyz[:, :, 2] + fu)
        ret[:, :, 1] = (flip * xyz[:, :, 1] * fy / xyz[:, :, 2] + fv)
        ret[:, :, 2] = xyz[:, :, 2]
    return ret


def offset2joint_weight(offset, depth, kernel_size):
    device = offset.device
    batch_size, joint_num, feature_size, feature_size = offset.size()
    joint_num = int(joint_num / 5)

    if depth.size(-1) != feature_size:
        depth = F.interpolate(depth, size=[feature_size, feature_size])

    offset_unit = offset[:, :joint_num * 3, :, :].contiguous()  # b * (3*J) * fs * fs
    heatmap = offset[:, joint_num * 3:joint_num * 4, :, :].contiguous()
    weight = offset[:, joint_num * 4:, :, :].contiguous()

    mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size,feature_size).float() + 0.5) / feature_size - 1.0
    mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size,feature_size).float() + 0.5) / feature_size - 1.0
    coords = torch.stack((mesh_y, mesh_x), dim=0)
    coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)

    coords = torch.cat((coords, depth), dim=1).repeat(1, joint_num, 1, 1).view(batch_size, joint_num, 3, -1)

    mask = depth.lt(0.99).float()
    offset_mask = (offset_unit * mask).view(batch_size, joint_num, 3, -1)
    heatmap_mask = (heatmap * mask).view(batch_size, joint_num, -1)
    weight_mask = weight.masked_fill(depth.gt(0.99), -1e8)
    normal_weight = F.softmax(weight_mask.view(batch_size, joint_num, -1), dim=-1)  # b * J * fs^2

    if torch.is_tensor(kernel_size):
        kernel_size = kernel_size.to(device)
        dist = kernel_size.view(1, joint_num, 1) - heatmap_mask * kernel_size.view(1, joint_num, 1)
    else:
        dist = kernel_size - heatmap_mask * kernel_size

    joint = torch.sum((offset_mask * dist.unsqueeze(2).repeat(1, 1, 3, 1) + coords) * normal_weight.unsqueeze(2).repeat(1, 1, 3, 1), dim=-1)
    return joint


def joint2offset(joint, img, feature_size,kernel_size=0.8):
    device = joint.device
    batch_size, _, img_height, img_width = img.size()
    img = F.interpolate(img,size=[feature_size,feature_size])
    _,joint_num,_ = joint.view(batch_size,-1,3).size()
    joint_feature = joint.reshape(joint.size(0),-1,1,1).repeat(1,1,feature_size,feature_size)
    # mesh_x = 2.0 * torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
    # mesh_y = 2.0 * torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() / (feature_size - 1.0) - 1.0
    mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
    mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).expand(feature_size, feature_size).float() + 0.5) / feature_size - 1.0
    coords = torch.stack((mesh_y, mesh_x), dim=0)
    coords = torch.unsqueeze(coords, dim=0).repeat(batch_size, 1, 1, 1).to(device)
    coords = torch.cat((coords, img), dim=1).repeat(1, joint_num, 1, 1)
    offset = joint_feature - coords
    offset = offset.view(batch_size, joint_num, 3, feature_size, feature_size)
    dist = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=2) + 1e-8)
    offset_norm = (offset / (dist.unsqueeze(2)))
    if torch.is_tensor(kernel_size):
        kernel_size = kernel_size.to(device)
        heatmap = (kernel_size.view(1, joint_num, 1, 1) - dist) / kernel_size.view(1, joint_num, 1, 1)
    else:
        heatmap = (kernel_size - dist)/kernel_size
    mask = heatmap.ge(0).float() * img.lt(0.99).float().view(batch_size,1,feature_size,feature_size)
    offset_norm_mask = (offset_norm * mask.unsqueeze(2)).view(batch_size, -1, feature_size, feature_size).float()
    heatmap_mask = heatmap * mask.float()
    return torch.cat((offset_norm_mask, heatmap_mask),dim=1)
    # return torch.cat((offset_norm.view(batch_size,-1,feature_size,feature_size), heatmap),dim=1).float()


def img2pcl_index(pcl, img, center, cube, cam_para, select_num=9,img_size=256):
    '''
    :param pcl: BxNx3 Tensor
    :param img: Bx1xWxH Tensor
    :param feature: BxCxWxH Tensor
    :return: select_feature: BxCxN
    '''

    device = pcl.device
    B, N, _ = pcl.size()
    B, _, W, H = img.size()

    mesh_x = 2.0 * (torch.arange(W).unsqueeze(1).expand(W, W).float() + 0.5) / W - 1.0
    mesh_y = 2.0 * (torch.arange(W).unsqueeze(0).expand(W, W).float() + 0.5) / W - 1.0
    coords = torch.stack((mesh_y, mesh_x), dim=0)
    coords = torch.unsqueeze(coords, dim=0).repeat(B, 1, 1, 1).to(device)
    img_uvd = torch.cat((coords, img), dim=1).view(B, 3, H * W).permute(0, 2, 1)
    img_xyz = uvd_nl2xyznl_tensor(img_uvd, center, cube, cam_para,img_size)

    # distance = torch.sqrt(torch.sum(torch.pow(pcl.unsqueeze(2) - img_xyz.unsqueeze(1), 2), dim=-1) + 1e-8)
    distance = torch.sum(torch.pow(pcl.unsqueeze(2) - img_xyz.unsqueeze(1), 2), dim=-1)
    distance_value, distance_index = torch.topk(distance, select_num, largest=False)
    # version 1
    closeness_value = 1 / (distance_value + 1e-8)
    closeness_value_normal = closeness_value / (closeness_value.sum(-1, keepdim=True) + 1e-8)

    # version 2
    # distance_value = torch.sqrt(distance_value + 1e-8)
    # distance_value = distance_value - distance_value.min(dim=-1,keepdim=True)[0]
    # closeness_value = 1 - distance_value / distance_value.max(dim=-1,keepdim=True)[0]
    # closeness_value_normal = torch.softmax(closeness_value*30, dim=-1)
    return closeness_value_normal, distance_index

def uvd_nl2xyznl_tensor(uvd, center, cube, cam_paras,img_size):
    batch_size, point_num, _ = uvd.size()
    device = uvd.device
    cube_size_t = cube.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
    center_t = center.to(device).view(batch_size, 1, 3).repeat(1, point_num, 1)
    #M_t = m.to(device).view(batch_size, 1, 3, 3)
    #M_inverse = torch.linalg.inv(M_t).repeat(1, point_num, 1, 1)

    uv_unnormal = (uvd[:, :, 0:2] + 1) * (img_size / 2)
    d_unnormal = (uvd[:, :, 2:]) * (cube_size_t[:, :, 2:] / 2.0) + center_t[:, :, 2:]
    uvd_unnormal = torch.cat((uv_unnormal, d_unnormal), dim=-1)
    #uvd_world = get_trans_points(uvd_unnormal, M_inverse)
    uvd_world = uvd_unnormal
    xyz = pointsImgTo3D(uvd_world, cam_paras)
    xyz_noraml = (xyz - center_t) / (cube_size_t / 2.0)
    return xyz_noraml

def get_trans_points(joints, M):
    device = joints.device
    joints_mat = torch.cat((joints[:, :, 0:2], torch.ones(joints.size(0), joints.size(1), 1).to(device)), dim=-1)
    joints_trans_xy = torch.matmul(M, joints_mat.unsqueeze(-1)).squeeze(-1)[:, :, 0:2]
    joints_trans_z = joints[:, :, 2:]
    return torch.cat((joints_trans_xy, joints_trans_z), dim=-1)

def pointsImgTo3D( point_uvd, paras, flip=1):
    point_xyz = torch.zeros_like(point_uvd).to(point_uvd.device)
    point_xyz[:, :, 0] = (point_uvd[:, :, 0] - paras[:, 0,2].unsqueeze(1)) * point_uvd[:, :, 2] / paras[:,0,0].unsqueeze(1)
    point_xyz[:, :, 1] = flip * (point_uvd[:, :, 1] - paras[:, 1,2].unsqueeze(1)) * point_uvd[:, :, 2] / paras[:,1,1].unsqueeze(1)
    point_xyz[:, :, 2] = point_uvd[:, :, 2]
    return point_xyz



def point2pcl_index(cfg,hand_center_3d,center_t,cube_size_t,xyz_points,pcl,K=4):
    '''
    :param pcl: BxNx3 Tensor
    :param xyz_points: BxMx3 Tensor
    '''
    xyz = xyz_points * 2 / cfg.recon_scale + hand_center_3d.unsqueeze(1)
    xyz = xyz * 1000  # m2mm
    xyz_noraml = (xyz - center_t.unsqueeze(1)) / (cube_size_t.unsqueeze(1) / 2.0)
    # distance = torch.sqrt(torch.sum(torch.pow(pcl.unsqueeze(2) - img_xyz.unsqueeze(1), 2), dim=-1) + 1e-8)
    distance = torch.sum(torch.pow(xyz_noraml.unsqueeze(2)-pcl.unsqueeze(1), 2), dim=-1)#BMN
    distance_value, distance_index = torch.topk(distance, K, largest=False)#BMK

    closeness_value = 1 / (distance_value + 1e-8)
    closeness_value_normal = closeness_value / (closeness_value.sum(-1, keepdim=True) + 1e-8)

    return closeness_value_normal, distance_index

def point2pcl_index_cylinder(cfg,hand_center_3d,center_t,cube_size_t,xyz_points,pcl,K=16,sigma=0.1):
    '''
    :param pcl: BxNx3 Tensor
    :param xyz_points: BxMx3 Tensor
    '''
    # transform

    xyz = xyz_points * 2 / cfg.recon_scale + hand_center_3d.unsqueeze(1)
    xyz = xyz * 1000  # m2mm

    xyz_noraml = (xyz - center_t.unsqueeze(1)) / (cube_size_t.unsqueeze(1) / 2.0)
    # distance = torch.sqrt(torch.sum(torch.pow(pcl.unsqueeze(2) - img_xyz.unsqueeze(1), 2), dim=-1) + 1e-8)
    distance = torch.sum(torch.pow(xyz_noraml.unsqueeze(2)-pcl.unsqueeze(1), 2), dim=-1)#BMN
    distance_xy = torch.sum(torch.pow(xyz_noraml[:,:,:2].unsqueeze(2)-pcl[:,:,:2].unsqueeze(1), 2), dim=-1)#BMN
    distance[distance_xy>=sigma] = 1e8

    distance_value, distance_index = torch.topk(distance, K, largest=False)#BMK

    closeness_value = 1 / (distance_value + 1e-8)
    closeness_value_normal = closeness_value / (closeness_value.sum(-1, keepdim=True) + 1e-8)

    return closeness_value_normal, distance_index


def point2pcl_index_sdf2weight(cfg,hand_center_3d,center_t,cube_size_t,xyz_points,pcl,sdf_value_hand,sdf_value_obj,sdf_factor,K=16,sigma=0.1):
    '''
    :param pcl: BxNx3 Tensor
    :param xyz_points: BxMx3 Tensor
    '''
    # transform
    B,M,_ = xyz_points.shape
    N=cfg.num_sample_points
    xyz = xyz_points * 2 / cfg.recon_scale + hand_center_3d.unsqueeze(1)

    xyz = xyz*1000 #m2mm
    xyz_noraml = (xyz - center_t.unsqueeze(1)) / (cube_size_t.unsqueeze(1) / 2.0)

    distance = torch.sum(torch.pow(xyz_noraml.unsqueeze(2)-pcl.unsqueeze(1), 2), dim=-1)#BMN
    distance_xy = torch.sum(torch.pow(xyz_noraml[:,:,:2].unsqueeze(2)-pcl[:,:,:2].unsqueeze(1), 2), dim=-1)#BMN

    # dis_xy
    distance_value_xy, distance_index = torch.topk(distance_xy, K, largest=False)  # BMK
    distance_value = torch.gather(distance,-1,distance_index)

    if cfg.sdf_weight:
        sdf_value_hand = torch.gather(sdf_value_hand.squeeze(-1).unsqueeze(1).repeat(1,M,1), -1, distance_index)
        sdf_value_hand[sdf_value_hand< 0]=0
        sdf_value_obj = torch.gather(sdf_value_obj.squeeze(-1).unsqueeze(1).repeat(1,M,1), -1, distance_index)
        sdf_value_obj[sdf_value_obj < 0] = 0
    else:
        sdf_value_hand = 0
        sdf_value_obj = 0
    assert cfg.sdf_weight is True

    closeness_value_hand = 1 / (distance_value + (sdf_value_hand * sdf_factor[0]) + torch.clamp(sdf_factor[2],min=1e-8))
    closeness_value_normal_hand = closeness_value_hand / (closeness_value_hand.sum(-1, keepdim=True) + 1e-8)
    closeness_value_obj = 1 / (distance_value + (sdf_value_obj * sdf_factor[1]) +torch.clamp(sdf_factor[2],min=1e-8))
    closeness_value_normal_obj = closeness_value_obj / (closeness_value_obj.sum(-1, keepdim=True) + 1e-8)

    return closeness_value_normal_hand,closeness_value_normal_obj, distance_index

def pcl2sample(pcl,center_t,cube_size_t,hand_center_3d,cfg):
    #xyz_unnoraml=(pcl - center_t.unsqueeze(1)) / (cube_size_t.unsqueeze(1) / 2.0)
    xyz_unnormal = (pcl*(cube_size_t.unsqueeze(1) / 2.0)+ center_t.unsqueeze(1))/1000
    xyz_sample = (xyz_unnormal - hand_center_3d.unsqueeze(1)) *cfg.recon_scale/2

    return xyz_sample

def sample2pcl(sample,center_t,cube_size_t,hand_center_3d,cfg):
    #xyz_unnoraml=(pcl - center_t.unsqueeze(1)) / (cube_size_t.unsqueeze(1) / 2.0)
    xyz = sample * 2 / cfg.recon_scale + hand_center_3d.unsqueeze(1)
    xyz = xyz * 1000  # m2mm
    xyz_noraml = (xyz - center_t.unsqueeze(1)) / (cube_size_t.unsqueeze(1) / 2.0)

    return xyz_noraml

