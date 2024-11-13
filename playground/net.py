#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import time

import torchvision
from torch.nn import functional as F

from networks.necks.unet import AWRUnet

from common.networks.backbones.resnet import load_dualpath_model, RGBD_ResNet, RGBD_BasicBlock
from common.utils.loss import SmoothL1Loss
from common.utils.pose_utils import joint3DToImg, offset2joint_weight, joint2offset, img2pcl_index, point2pcl_index, \
    point2pcl_index_cylinder, point2pcl_index_sdf2weight, pcl2sample, sample2pcl
from common.utils.sdf_utils import pixel_align
from config import cfg
from networks.backbones.resnet import ResNetBackbone,ResNet
from networks.necks.unet import UNet
from networks.heads.sdf_head import SDFHead
from networks.heads.mano_head import ManoHead
from networks.heads.fc_head import FCHead
from mano.mano_preds import get_mano_preds
from mano.manolayer import ManoLayer
from mano.inverse_kinematics import ik_solver_mano
from utils.pose_utils import soft_argmax, decode_volume
from utils.sdf_utils import kinematic_embedding


class model(nn.Module):
    def __init__(self, cfg, backbone, neck, hand_sdf_head, obj_sdf_head, mano_head, volume_head, rot_head,awr_head,heatmap_Unet,heatmap_head):
        super(model, self).__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.dim_backbone_feat = 2048 if self.cfg.backbone == 'resnet_50' else 512
        self.neck = neck
        self.heatmap_Unet = heatmap_Unet
        self.hand_sdf_head = hand_sdf_head
        self.obj_sdf_head = obj_sdf_head
        self.mano_head = mano_head
        self.volume_head = volume_head
        self.heatmap_head = heatmap_head
        self.rot_head = rot_head
        self.awr = cfg.awr_branch
        self.pcl_embedding = cfg.pcl_branch

        if self.mano_head is not None:
            self.backbone_2_mano = nn.Sequential(
                nn.Conv2d(in_channels=self.dim_backbone_feat, out_channels=512, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True))

        if self.rot_head is not None:
            self.backbone_2_rot = nn.Sequential(
                nn.Conv2d(in_channels=self.dim_backbone_feat, out_channels=512, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True))

        if self.awr:
            self.dim_backbone_feat =self.dim_backbone_feat// 4
            self.awr_head = awr_head
        if self.pcl_embedding:
            self.pcl_feat_emb = nn.Sequential(nn.Conv1d(self.dim_backbone_feat, self.cfg.sdf_latent, 1), nn.BatchNorm1d(self.cfg.sdf_latent))
            self.pcl_xyz_emb = nn.Sequential(nn.Conv1d(3, self.cfg.sdf_latent, 1), nn.BatchNorm1d(self.cfg.sdf_latent))
            self.pcl_sdf_emb = nn.Sequential(nn.Conv1d(2, self.cfg.sdf_latent, 1), nn.BatchNorm1d(self.cfg.sdf_latent))
        if self.cfg.heatmap:
            self.fc_expand2voxel = nn.Linear(self.cfg.pcl_sample_nums, 32*32*32)
            self.fc_pred_heatmap =nn.Sequential(
                nn.Linear(self.cfg.sdf_latent, self.cfg.sdf_latent // 4),
                nn.ReLU(inplace=True),
                nn.Linear(self.cfg.sdf_latent // 4, 2),
                nn.Sigmoid()
            )

        self.hand_sdf_head_global = SDFHead(cfg.sdf_latent, cfg.hand_point_latent, cfg.sdf_head['dims'],
                                cfg.sdf_head['dropout'], cfg.sdf_head['dropout_prob'],
                                cfg.sdf_head['norm_layers'], cfg.sdf_head['latent_in'], cfg.hand_cls,
                                cfg.sdf_head['num_class'])

        self.obj_sdf_head_global = SDFHead(cfg.sdf_latent, cfg.obj_point_latent, cfg.sdf_head['dims'],
                               cfg.sdf_head['dropout'], cfg.sdf_head['dropout_prob'],
                               cfg.sdf_head['norm_layers'], cfg.sdf_head['latent_in'], False,
                               cfg.sdf_head['num_class'])


        self.backbone_2_sdf = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_backbone_feat, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.backbone_2_sdf_global = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_backbone_feat, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        if self.cfg.with_add_feats:
            self.sdf_encoder = nn.Linear(516, self.cfg.sdf_latent)
            self.sdf_encoder_global = nn.Linear(512, self.cfg.sdf_latent)
        else:
            self.sdf_encoder = nn.Linear(512, self.cfg.sdf_latent)
            self.sdf_encoder_global = nn.Linear(512, self.cfg.sdf_latent)

        self.loss_l1 = torch.nn.L1Loss(reduction='sum')
        self.smoothL1Loss = SmoothL1Loss().cuda()
        self.loss_l2 = torch.nn.MSELoss()
        self.loss_ce = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.sigma = nn.Parameter(torch.tensor([0.3]))

        self.sdf_factor = nn.Parameter(torch.zeros([4]))

    #cond_input may include camera intrinsics or hand wrist position
    def forward(self, inputs, targets=None, metas=None, mode='train',writer=None,steps=0):
        if mode == 'train':
            input_img = inputs['img']
            img_depth = inputs['img_d']
            if self.cfg.hand_branch and self.cfg.obj_branch:
                sdf_data = torch.cat([targets['hand_sdf'], targets['obj_sdf']], 1)
                cls_data = torch.cat([targets['hand_labels'], targets['obj_labels']], 1)
                if metas['epoch'] < self.cfg.sdf_add_epoch:
                    mask_hand = torch.cat([torch.ones(targets['hand_sdf'].size()[:2]), torch.zeros(targets['obj_sdf'].size()[:2])], 1)
                    mask_hand = (mask_hand.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
                    mask_obj = torch.cat([torch.zeros(targets['hand_sdf'].size()[:2]), torch.ones(targets['obj_sdf'].size()[:2])], 1)
                    mask_obj = (mask_obj.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
                else:
                    mask_hand = torch.cat([torch.ones(targets['hand_sdf'].size()[:2]), torch.ones(targets['obj_sdf'].size()[:2])], 1)
                    mask_hand = (mask_hand.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
                    mask_obj = torch.cat([torch.ones(targets['hand_sdf'].size()[:2]), torch.ones(targets['obj_sdf'].size()[:2])], 1)
                    mask_obj = (mask_obj.cuda()).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1)
            elif self.cfg.hand_branch:
                sdf_data = targets['hand_sdf']
                cls_data = targets['hand_labels']
                mask_hand = torch.ones(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1).cuda()
            elif self.cfg.obj_branch:
                sdf_data = targets['obj_sdf']
                cls_data = targets['obj_labels']
                mask_obj = torch.ones(self.cfg.train_batch_size * self.cfg.num_sample_points).unsqueeze(1).cuda()

            sdf_data = sdf_data.reshape(self.cfg.train_batch_size * self.cfg.num_sample_points, -1)
            cls_data = cls_data.to(torch.long).reshape(self.cfg.train_batch_size * self.cfg.num_sample_points)
            xyz_points = sdf_data[:, 0:-2] #(B*2000)*3
            sdf_gt_hand = sdf_data[:, -2].unsqueeze(1)
            sdf_gt_obj = sdf_data[:, -1].unsqueeze(1)
            if self.cfg.hand_branch:
                sdf_gt_hand = torch.clamp(sdf_gt_hand, -self.cfg.clamp_dist, self.cfg.clamp_dist)
            if self.cfg.obj_branch:
                sdf_gt_obj = torch.clamp(sdf_gt_obj, -self.cfg.clamp_dist, self.cfg.clamp_dist)

            # go through backbone
            if self.cfg.modal =='RGBD':
                _, merges = self.backbone(input_img,img_depth,writer,steps)
                c1, c2, c3,c4 = merges
            else:
                _,c1, c2, c3,c4 = self.backbone(input_img)

            volume_results = {}
            if self.volume_head is not None:
                hm_feat = self.neck(c4)
                hm_pred = self.volume_head(hm_feat)
                hm_pred, hm_conf = soft_argmax(cfg, hm_pred, num_joints=1)
                volume_joint_preds = decode_volume(cfg, hm_pred, metas['hand_center_3d'], metas['cam_intr'])
                volume_results['joints'] = volume_joint_preds
            else:
                volume_results = None

            if self.heatmap_Unet is not None:
                hm_feat_ho = self.heatmap_Unet(c4)
                hm_pred_ho = self.heatmap_head(hm_feat_ho).reshape(input_img.shape[0],2,32,32,32)
            else:
                hm_pred_ho = None

            # estimate the hand pose
            if self.mano_head is not None:
                mano_feat = self.backbone_2_mano(c4)
                mano_feat = mano_feat.mean(3).mean(2)
                hand_pose_results = self.mano_head(mano_feat)
                hand_pose_results = get_mano_preds(hand_pose_results, self.cfg, metas['cam_intr'], metas['hand_center_3d'])
            else:
                hand_pose_results = None
        
            # estimate the object rotation
            if self.rot_head is not None:
                rot_feat = self.backbone_2_rot(c4)
                rot_feat = rot_feat.mean(3).mean(2)
                obj_rot = self.rot_head(rot_feat)
            else:
                obj_rot = None
        
            # convert the object pose to the hand-relative coordinate system
            if cfg.obj_trans or cfg.obj_rot:
                obj_pose_results = {}
                obj_transform = torch.zeros(self.cfg.train_batch_size, 4, 4).to(input_img.device)
                obj_transform[:, :3, 3] = volume_joint_preds[:, 0, :] - metas['hand_center_3d']
                obj_transform[:, 3, 3] = 1
                if cfg.obj_rot:
                    obj_transform[:, :3, :3] = obj_rot
                    obj_corners = torch.matmul(obj_rot, metas['obj_rest_corners_3d']).transpose(2, 1).transpose(2, 1) + volume_joint_preds
                else:
                    obj_transform[:, :3, :3] = torch.eye(3).to(input_img.device)
                    obj_corners = metas['obj_rest_corners_3d'] + volume_joint_preds
                obj_pose_results['global_trans'] = obj_transform
                obj_pose_results['center'] = volume_joint_preds
                obj_pose_results['corners'] = obj_corners
                if hand_pose_results is not None:
                    obj_pose_results['wrist_trans'] = hand_pose_results['global_trans'][:, 0]
            else:
                obj_pose_results = None

            if self.awr:
                init_pose_result,backbone_feat = self.awr_head(c1, c2, c3,c4)

            pcl = inputs['pcl']
            B, N, _ = pcl.size()

            # generate features for the sdf head

            sdf_feat = self.backbone_2_sdf(backbone_feat)
            sdf_feat_local, _ = pixel_align(self.cfg, xyz_points, self.cfg.num_sample_points, sdf_feat,
                                      metas['hand_center_3d'], metas['cam_intr'])
            sdf_feat = self.sdf_encoder(sdf_feat_local)

            #sdf_feat_global = self.backbone_2_sdf_global(c4)
            sdf_feat_global = c4
            sdf_feat_global = sdf_feat_global.mean(3).mean(2)
            sdf_feat_global = self.sdf_encoder_global(sdf_feat_global)
            sdf_feat_global_forPCL = sdf_feat_global.clone()
            sdf_feat_global_forPCL = sdf_feat_global_forPCL.repeat_interleave(N, dim=0)
            sdf_feat_global = sdf_feat_global.repeat_interleave(self.cfg.num_sample_points, dim=0)


            # Stage1, with global feature, and then classify the PCL to hand/ocj/bg PCL
            if self.hand_sdf_head_global is not None:
                #hand_points = sample2pcl(xyz_points.reshape(B,self.cfg.num_sample_points,3),metas['center_xyz'], metas['cube'], metas['hand_center_3d'], self.cfg)
                hand_points = xyz_points.reshape((-1, self.cfg.hand_point_latent))
                # point_feature
                #hand_sdf_decoder_inputs = torch.cat([sdf_feat_global, point_feat, hand_points], dim=1)
                # global feature Only
                hand_sdf_global_inputs = torch.cat([sdf_feat_global, hand_points], dim=1)

                sdf_hand_global, cls_hand_global = self.hand_sdf_head_global(hand_sdf_global_inputs)
                sdf_hand_global = torch.clamp(sdf_hand_global, min=-self.cfg.clamp_dist, max=self.cfg.clamp_dist)
            else:
                sdf_hand_global = None
                cls_hand_global = None

            if self.obj_sdf_head_global is not None:
                #obj_points = sample2pcl(xyz_points.reshape(B,self.cfg.num_sample_points,3),metas['center_xyz'], metas['cube'], metas['hand_center_3d'], self.cfg)
                obj_points = xyz_points.reshape((-1, self.cfg.obj_point_latent))
                # point_feature
                #obj_sdf_decoder_inputs = torch.cat([sdf_feat_global, point_feat, obj_points], dim=1)
                obj_sdf_global_inputs = torch.cat([sdf_feat_global, obj_points], dim=1)
                sdf_obj_global, _ = self.obj_sdf_head_global(obj_sdf_global_inputs)
                sdf_obj_global = torch.clamp(sdf_obj_global, min=-self.cfg.clamp_dist, max=self.cfg.clamp_dist)
            else:
                sdf_obj_global = None
            pcl_sample = pcl2sample(pcl, metas['center_xyz'], metas['cube'], metas['hand_center_3d'], self.cfg)

            pcl_classify_inputs = torch.cat([sdf_feat_global_forPCL, pcl_sample.reshape(-1,3)], dim=1)
            pcl_classify_hand,_ = self.hand_sdf_head_global(pcl_classify_inputs)
            pcl_classify_hand = torch.clamp(pcl_classify_hand, min=-self.cfg.clamp_dist, max=self.cfg.clamp_dist).reshape(B,N,1)
            pcl_classify_obj, _ = self.obj_sdf_head_global(pcl_classify_inputs)
            pcl_classify_obj = torch.clamp(pcl_classify_obj, min=-self.cfg.clamp_dist, max=self.cfg.clamp_dist).reshape(B,N,1)

            # pcl_classify_hand = pcl_classify_hand.detach()
            # pcl_classify_obj = pcl_classify_obj.detach()

            if self.pcl_embedding:
                img_feat = backbone_feat.clone()
                B, C, H, W = img_feat.size()
                img_down = F.interpolate(img_depth, [H, W])
                pcl_closeness, pcl_index = img2pcl_index(pcl, img_down, metas['center_xyz'], metas['cube'], metas['cam_intr'], select_num=4,img_size=self.cfg.image_size[0])
                pcl_feat_index = pcl_index.view(B, 1, -1).repeat(1, C, 1)  # B*128*(K*1024)
                pcl_feat = torch.gather(img_feat.view(B, C, -1), -1, pcl_feat_index).view(B, C, N, -1)
                pcl_feat = torch.sum(pcl_feat * pcl_closeness.unsqueeze(1), dim=-1).permute(0, 2, 1)

                pcl_feat = self.pcl_feat_emb(pcl_feat.permute(0, 2, 1)).permute(0, 2, 1) + \
                           self.pcl_xyz_emb(pcl.permute(0, 2, 1)).permute(0, 2, 1) + \
                           self.pcl_sdf_emb(torch.cat((pcl_classify_hand,pcl_classify_obj),dim=-1).permute(0, 2, 1)).permute(0, 2, 1)

                pcl_feat = F.relu(pcl_feat)

                sdf_factor = torch.sigmoid(self.sdf_factor)

                #point_closeness, point_index = point2pcl_index_cylinder(self.cfg,metas['hand_center_3d'],metas['center_xyz'], metas['cube'],xyz_points.reshape(B, self.cfg.num_sample_points, 3),pcl,K=16,sigma=self.sigma*self.sigma)
                point_closeness_hand, point_closeness_obj,point_index = point2pcl_index_sdf2weight(self.cfg, metas['hand_center_3d'],
                                                                        metas['center_xyz'], metas['cube'],
                                                                        xyz_points.reshape(B,self.cfg.num_sample_points,3),pcl,
                                                                        pcl_classify_hand,pcl_classify_obj,sdf_factor,
                                                                        K=self.cfg.K,
                                                                        sigma=self.sigma * self.sigma)

                point_feat_index = point_index.view(B, 1, -1).repeat(1, C, 1)  # B*128*(K*M)

                point_feat = torch.gather(pcl_feat.permute(0, 2, 1), -1, point_feat_index).view(B, C,self.cfg.num_sample_points, -1)
                point_feat_hand = torch.sum(point_feat * point_closeness_hand.unsqueeze(1), dim=-1).permute(0, 2, 1)
                point_feat_hand = point_feat_hand.reshape(B*self.cfg.num_sample_points, C)
                point_feat_obj = torch.sum(point_feat * point_closeness_obj.unsqueeze(1), dim=-1).permute(0, 2, 1)
                point_feat_obj = point_feat_obj.reshape(B * self.cfg.num_sample_points, C)
                if steps%10==0:
                    writer.add_scalar('sdf_factor_hand',sdf_factor[0],steps)
                    writer.add_scalar('sdf_factor_obj', sdf_factor[1], steps)
                    writer.add_scalar('sdf_offset_hand', sdf_factor[2], steps)
                    writer.add_scalar('sdf_offset_obj', sdf_factor[3], steps)
                    writer.add_scalar('pcl_classify_hand', pcl_classify_hand.min(), steps)
                    writer.add_scalar('pcl_classify_obj', pcl_classify_obj.min(), steps)
                    writer.add_scalar('pcl_classify_hand_mean', pcl_classify_hand.mean(), steps)
                    writer.add_scalar('sample_mean', hand_points[:,0].mean(), steps)
                    writer.add_scalar('pcl_sample_mean',pcl_sample[:,0].mean(), steps)
                    writer.add_scalar('pcl_mean', pcl[:, 0].mean(), steps)
                    writer.add_scalar('sample_max', hand_points[:, 0].max(), steps)
                    writer.add_scalar('pcl_sample_max', pcl_sample[:, 0].max(), steps)
                    writer.add_scalar('pcl_max', pcl[:, 0].max(), steps)

            # Stage2, with pixel-aligned feature
            if self.hand_sdf_head is not None:
                if self.cfg.hand_encode_style == 'kine':
                    if metas['epoch'] >= self.cfg.pose_epoch:
                        hand_points = kinematic_embedding(self.cfg, xyz_points, self.cfg.num_sample_points, hand_pose_results, 'hand')
                        hand_points = hand_points.reshape((-1, self.cfg.hand_point_latent))
                    else:
                        mano_layer = ManoLayer(ncomps=45, center_idx=0, side="right", mano_root='../common/mano/assets/', use_pca=False, flat_hand_mean=True).cuda()
                        _, _, _, gt_global_trans, gt_rot_center = mano_layer(metas['hand_poses'], th_betas=metas['hand_shapes'], root_palm=False)
                        gt_hand_pose_results = {}
                        gt_hand_pose_results['global_trans'] = gt_global_trans
                        hand_points = kinematic_embedding(self.cfg, xyz_points, self.cfg.num_sample_points, gt_hand_pose_results, 'hand')
                        hand_points = hand_points.reshape((-1, self.cfg.hand_point_latent))
                elif self.cfg.hand_encode_style == 'gt_kine':
                        mano_layer = ManoLayer(ncomps=45, center_idx=0, side="right", mano_root='../common/mano/assets/', use_pca=False, flat_hand_mean=True).cuda()
                        _, _, _, gt_global_trans, gt_rot_center = mano_layer(metas['hand_poses'], th_betas=metas['hand_shapes'], root_palm=False)
                        gt_hand_pose_results = {}
                        gt_hand_pose_results['global_trans'] = gt_global_trans
                        hand_points = kinematic_embedding(self.cfg, xyz_points, self.cfg.num_sample_points, gt_hand_pose_results, 'hand')
                        hand_points = hand_points.reshape((-1, self.cfg.hand_point_latent))
                else:
                    hand_points = xyz_points.reshape((-1, self.cfg.hand_point_latent))
                # point_feature
                if self.pcl_embedding:
                    #hand_sdf_decoder_inputs = torch.cat([sdf_feat, point_feat, hand_points], dim=1)
                    hand_sdf_decoder_inputs = torch.cat([sdf_feat, point_feat_hand, hand_points], dim=1)
                else:
                    hand_sdf_decoder_inputs = torch.cat([sdf_feat, hand_points], dim=1)
                sdf_hand, cls_hand = self.hand_sdf_head(hand_sdf_decoder_inputs)
                sdf_hand = torch.clamp(sdf_hand, min=-self.cfg.clamp_dist, max=self.cfg.clamp_dist)
            else:
                sdf_hand = None
                cls_hand = None
        
            if self.obj_sdf_head is not None:
                if self.cfg.obj_encode_style == 'kine':
                    if metas['epoch'] >= self.cfg.pose_epoch: 
                        obj_points = kinematic_embedding(self.cfg, xyz_points, self.cfg.num_sample_points, obj_pose_results, 'obj')
                        obj_points = obj_points.reshape((-1, self.cfg.obj_point_latent))
                    else:
                        gt_obj_pose_results = {}
                        gt_obj_pose = metas['obj_transform']
                        if self.rot_head is None:
                            gt_obj_pose[:, :3, :3] = torch.eye(3)
                        gt_obj_pose_results['global_trans'] = gt_obj_pose
                        obj_points = kinematic_embedding(self.cfg, xyz_points, self.cfg.num_sample_points, gt_obj_pose_results, 'obj')
                        obj_points = obj_points.reshape((-1, self.cfg.obj_point_latent))
                elif self.cfg.obj_encode_style == 'gt_trans':
                        gt_obj_pose_results = {}
                        gt_obj_pose = metas['obj_transform']
                        gt_obj_pose[:, :3, :3] = torch.eye(3)
                        gt_obj_pose_results['global_trans'] = gt_obj_pose
                        obj_points = kinematic_embedding(self.cfg, xyz_points, self.cfg.num_sample_points, gt_obj_pose_results, 'obj')
                        obj_points = obj_points.reshape((-1, self.cfg.obj_point_latent))
                elif self.cfg.obj_encode_style == 'gt_transrot':
                        gt_obj_pose_results = {}
                        gt_obj_pose = metas['obj_transform']
                        gt_obj_pose_results['global_trans'] = gt_obj_pose
                        obj_points = kinematic_embedding(self.cfg, xyz_points, self.cfg.num_sample_points, gt_obj_pose_results, 'obj')
                        obj_points = obj_points.reshape((-1, self.cfg.obj_point_latent))
                else:
                    obj_points = xyz_points.reshape((-1, self.cfg.obj_point_latent))
                # point_feature
                if self.pcl_embedding:
                    # point_closeness, point_index = point2pcl_index(pcl, xyz_points, K=4)
                    # point_feat_index = point_index.view(B, 1, -1).repeat(1, C, 1)  # B*128*K
                    # point_feat = torch.gather(pcl_feat, -1, point_feat_index).view(B, C, -1)  # B*128*K
                    # point_feat = torch.sum(point_feat * point_closeness.unsqueeze(1), dim=-1).permute(0, 2, 1)
                    obj_sdf_decoder_inputs = torch.cat([sdf_feat, point_feat_obj,obj_points], dim=1)
                else:
                    obj_sdf_decoder_inputs = torch.cat([sdf_feat, obj_points], dim=1)
                sdf_obj, _ = self.obj_sdf_head(obj_sdf_decoder_inputs)
                sdf_obj = torch.clamp(sdf_obj, min=-self.cfg.clamp_dist, max=self.cfg.clamp_dist)
            else:
                sdf_obj = None

            sdf_results = {}
            sdf_results['hand'] = sdf_hand
            sdf_results['obj'] = sdf_obj
            sdf_results['cls'] = cls_hand

            loss = {}
            if self.hand_sdf_head is not None:
                loss['hand_sdf'] = self.cfg.hand_sdf_weight * self.loss_l1(sdf_hand * mask_hand, sdf_gt_hand * mask_hand) / mask_hand.sum()
                loss['hand_sdf_global'] = self.cfg.hand_sdf_weight * self.loss_l1(sdf_hand_global * mask_hand,
                                                                           sdf_gt_hand * mask_hand) / mask_hand.sum()

            if self.obj_sdf_head is not None:
                loss['obj_sdf'] = self.cfg.obj_sdf_weight * self.loss_l1(sdf_obj * mask_obj, sdf_gt_obj * mask_obj) / mask_obj.sum()
                loss['obj_sdf_global'] = self.cfg.obj_sdf_weight * self.loss_l1(sdf_obj_global * mask_obj,
                                                                         sdf_gt_obj * mask_obj) / mask_obj.sum()


            if cls_hand is not None:
                if metas['epoch'] >= self.cfg.sdf_add_epoch:
                    loss['hand_cls'] = self.cfg.hand_cls_weight * self.loss_ce(cls_hand, cls_data)
                else:
                    loss['hand_cls'] = 0. * self.loss_ce(cls_hand, cls_data)

            if self.mano_head is not None:
                valid_idx = hand_pose_results['vis']
                loss['mano_joints'] = self.cfg.mano_joints_weight * self.loss_l2(valid_idx.unsqueeze(-1) * hand_pose_results['joints'], valid_idx.unsqueeze(-1) * targets['hand_joints_3d'])
                loss['mano_shape'] = self.cfg.mano_shape_weight * self.loss_l2(hand_pose_results['shape'], torch.zeros_like(hand_pose_results['shape']))
                loss['mano_pose'] = self.cfg.mano_pose_weight * self.loss_l2(valid_idx * (hand_pose_results['pose'][:, 3:] - hand_pose_results['mean_pose']), valid_idx * torch.zeros_like(hand_pose_results['pose'][:, 3:]))

            if self.volume_head is not None:
                volume_joint_targets = targets['obj_center_3d'].unsqueeze(1)
                loss['volume_joint'] = self.cfg.volume_weight * self.loss_l2(volume_joint_preds, volume_joint_targets)
            
            if self.rot_head is not None:
                loss['obj_corner'] = self.cfg.corner_weight * self.loss_l2(obj_pose_results['corners'], targets['obj_corners_3d'])

            if self.heatmap_Unet is not None:
                loss['heatmap_ho'] = self.cfg.heatmap_weight * self.loss_l2(hm_pred_ho, targets['hm_ho'])
                #loss['heatmap_ho_xyz'] = self.cfg.heatmap_weight * self.loss_l2(heatmap_ho_xyz, targets['hm_ho'])

            if self.awr and self.cfg.pixel_weight!=0 and self.cfg.coord_weight!=0:
                #pixel_loss
                #xyz_gt = targets['hand_joints_3d']
                joint_img = metas['joint_img']
                # joint_img = joint3DToImg(xyz_gt*1000, metas['cam_intr'])
                # joint_img[:,:, 0:2] = joint_img[:,:,0:2] / (self.cfg.image_size[0] / 2) - 1
                # joint_img[:,:, 2] = (joint_img[:,:, 2] - metas['center_xyz'][:,2]) / (metas['cube'][:,2] / 2.0)

                joint_uvd = offset2joint_weight(init_pose_result, img_depth, kernel_size=0.8)
                feature_size = init_pose_result.size(-1)
                pixel_gt = joint2offset(joint_img, img_depth, feature_size)
                loss['loss_pixel'] = self.smoothL1Loss(init_pose_result[:, :pixel_gt.size(1)],pixel_gt) * self.cfg.pixel_weight
                loss['loss_coord'] = self.smoothL1Loss(joint_uvd, joint_img) * self.cfg.coord_weight
                # loss += (loss_pixel*self.cfg.pixel_weight + loss_coord*self.cfg.coord_weight)

                # joint_loss
                # xyz_gt = targets['hand_joints_3d']
                # pixel_pd = init_pose_result
                # joint_uvd = offset2joint_weight(pixel_pd, img_depth, kernel)
            if self.cfg.heatmap:
                return loss, sdf_results, hand_pose_results, obj_pose_results, hm_pred_ho
            return loss, sdf_results, hand_pose_results, obj_pose_results

        else:
            with torch.no_grad():
                input_img = inputs['img']
                img_depth = inputs['img_d']
                # go through backbone
                if self.cfg.modal == 'RGBD':
                    _, merges = self.backbone(input_img, img_depth, writer, steps)
                    c1, c2, c3, c4 = merges
                else:
                    _, c1, c2, c3, c4 = self.backbone(input_img)

                volume_results = {}
                if self.volume_head is not None:
                    hm_feat = self.neck(c4)
                    hm_pred = self.volume_head(hm_feat)
                    hm_pred, hm_conf = soft_argmax(cfg, hm_pred, num_joints=1)
                    volume_joint_preds = decode_volume(cfg, hm_pred, metas['hand_center_3d'], metas['cam_intr'])
                    volume_results['joints'] = volume_joint_preds
                else:
                    volume_results = None

                if self.heatmap_Unet is not None:
                    hm_feat_ho = self.heatmap_Unet(c4)
                    hm_pred_ho = self.heatmap_head(hm_feat_ho).reshape(input_img.shape[0], 2, 32, 32, 32)
                else:
                    hm_pred_ho = None

                if self.mano_head is not None:
                    mano_feat = self.backbone_2_mano(c4)
                    mano_feat = mano_feat.mean(3).mean(2)
                    hand_pose_results = self.mano_head(mano_feat)
                    hand_pose_results = get_mano_preds(hand_pose_results, self.cfg, metas['cam_intr'], metas['hand_center_3d'])
                else:
                    hand_pose_results = None
        
                if self.rot_head is not None:
                    rot_feat = self.backbone_2_rot(c4)
                    rot_feat = rot_feat.mean(3).mean(2)
                    obj_rot = self.rot_head(rot_feat)
                else:
                    obj_rot = None

                # convert the object pose to the hand-relative coordinate system
                if cfg.obj_trans or cfg.obj_rot:
                    obj_pose_results = {}
                    obj_transform = torch.zeros(self.cfg.test_batch_size, 4, 4).to(input_img.device)
                    obj_transform[:, :3, 3] = volume_joint_preds[:, 0, :] - metas['hand_center_3d']
                    obj_transform[:, 3, 3] = 1
                    if cfg.obj_rot:
                        obj_transform[:, :3, :3] = obj_rot
                        obj_corners = torch.matmul(obj_rot, metas['obj_rest_corners_3d']).transpose(2, 1).transpose(2, 1) + volume_joint_preds
                    else:
                        obj_transform[:, :3, :3] = torch.eye(3).to(input_img.device)
                        obj_corners = metas['obj_rest_corners_3d'] + volume_joint_preds
                    obj_pose_results['global_trans'] = obj_transform
                    obj_pose_results['center'] = volume_joint_preds
                    obj_pose_results['corners'] = obj_corners
                    if hand_pose_results is not None:
                        obj_pose_results['wrist_trans'] = hand_pose_results['global_trans'][:, 0]
                else:
                    obj_pose_results = None

                if self.awr:
                    init_pose_result, backbone_feat = self.awr_head(c1, c2, c3, c4)

                pcl = inputs['pcl']
                B, N, _ = pcl.size()

                #sdf_feat_global = self.backbone_2_sdf_global(c4)
                sdf_feat_global=c4
                sdf_feat_global = sdf_feat_global.mean(3).mean(2)
                sdf_feat_global = self.sdf_encoder_global(sdf_feat_global)
                sdf_feat_global_forPCL = sdf_feat_global.clone()
                sdf_feat_global_forPCL = sdf_feat_global_forPCL.repeat_interleave(N, dim=0)

                pcl_sample = pcl2sample(pcl, metas['center_xyz'], metas['cube'], metas['hand_center_3d'], self.cfg)

                pcl_classify_inputs = torch.cat([sdf_feat_global_forPCL, pcl_sample.reshape(-1, 3)], dim=1)
                pcl_classify_hand,_ = self.hand_sdf_head_global(pcl_classify_inputs)
                pcl_classify_hand = torch.clamp(pcl_classify_hand, min=-self.cfg.clamp_dist,
                                                max=self.cfg.clamp_dist).reshape(B, N, 1)
                pcl_classify_obj,_ = self.obj_sdf_head_global(pcl_classify_inputs)
                pcl_classify_obj = torch.clamp(pcl_classify_obj, min=-self.cfg.clamp_dist,
                                               max=self.cfg.clamp_dist).reshape(B, N, 1)

                if self.pcl_embedding:

                    B, C, H, W = backbone_feat.size()
                    img_down = F.interpolate(img_depth, [H, W])
                    pcl_closeness, pcl_index = img2pcl_index(pcl, img_down, metas['center_xyz'], metas['cube'],
                                                             metas['cam_intr'], select_num=4,
                                                             img_size=self.cfg.image_size[0])
                    pcl_feat_index = pcl_index.view(B, 1, -1).repeat(1, C, 1)  # B*128*(K*1024)
                    pcl_feat = torch.gather(backbone_feat.view(B, C, -1), -1, pcl_feat_index).view(B, C, N, -1)
                    pcl_feat = torch.sum(pcl_feat * pcl_closeness.unsqueeze(1), dim=-1).permute(0, 2, 1)

                    pcl_feat = self.pcl_feat_emb(pcl_feat.permute(0, 2, 1)).permute(0, 2, 1) + \
                               self.pcl_xyz_emb(pcl.permute(0, 2, 1)).permute(0, 2, 1)+\
                               self.pcl_sdf_emb(torch.cat((pcl_classify_hand,pcl_classify_obj),dim=-1).permute(0, 2, 1)).permute(0, 2, 1)
                    pcl_feat = F.relu(pcl_feat)



            return backbone_feat, pcl_feat, pcl_classify_hand, pcl_classify_obj, hand_pose_results, obj_pose_results, hm_pred_ho


def get_model(cfg, is_train):
    num_resnet_layers = int(cfg.backbone.split('_')[-1])
    if cfg.modal=='RGBD':
        backbone = RGBD_ResNet(RGBD_BasicBlock, [2, 2, 2, 2])
        if is_train:
            print('load Pretrained_weight from resnet-18')
            pretrain_weight = torchvision.models.resnet18(pretrained=True)
            backbone = load_dualpath_model(backbone,pretrain_weight.state_dict())
        backbone.depth_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        backbone = ResNet(num_resnet_layers)
        if is_train:
            backbone.init_weights()

    neck_inplanes = 2048 if num_resnet_layers == 50 else 512
    if cfg.obj_trans:
        neck = UNet(neck_inplanes, 256, 3)
    else:
        neck = None

    if cfg.awr_branch:
        awr_head = AWRUnet(cfg.number_joints)
    else:
        awr_head = None

    if cfg.hand_branch:
        if cfg.pcl_branch:
            hand_sdf_head = SDFHead(cfg.sdf_latent*2, cfg.hand_point_latent, cfg.sdf_head['dims'],
                                    cfg.sdf_head['dropout'], cfg.sdf_head['dropout_prob'], cfg.sdf_head['norm_layers'],
                                    cfg.sdf_head['latent_in'], cfg.hand_cls, cfg.sdf_head['num_class'])
        else:
            hand_sdf_head = SDFHead(cfg.sdf_latent, cfg.hand_point_latent, cfg.sdf_head['dims'], cfg.sdf_head['dropout'], cfg.sdf_head['dropout_prob'], cfg.sdf_head['norm_layers'], cfg.sdf_head['latent_in'], cfg.hand_cls, cfg.sdf_head['num_class'])
    else:
        hand_sdf_head = None
    
    if cfg.obj_branch:
        if cfg.pcl_branch:
            obj_sdf_head = SDFHead(cfg.sdf_latent*2, cfg.obj_point_latent, cfg.sdf_head['dims'], cfg.sdf_head['dropout'], cfg.sdf_head['dropout_prob'], cfg.sdf_head['norm_layers'], cfg.sdf_head['latent_in'], False, cfg.sdf_head['num_class'])
        else:
            obj_sdf_head = SDFHead(cfg.sdf_latent, cfg.obj_point_latent, cfg.sdf_head['dims'], cfg.sdf_head['dropout'], cfg.sdf_head['dropout_prob'], cfg.sdf_head['norm_layers'], cfg.sdf_head['latent_in'], False, cfg.sdf_head['num_class'])
    else:
        obj_sdf_head = None
    
    if cfg.mano_branch:
        mano_head = ManoHead(depth=cfg.mano_depth)
    else:
        mano_head = None
    
    if cfg.obj_trans:
        volume_head = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)
    else:
        volume_head = None
    
    if cfg.obj_rot:
        rot_head = RotHead(rot_format=cfg.rot_format)
    else:
        rot_head = None

    if cfg.heatmap:
        heatmap_head = nn.Conv2d(in_channels=256, out_channels=32*2, kernel_size=1, stride=1, padding=0)
        heatmap_Unet = UNet(neck_inplanes, 256, 2)
    else:
        heatmap_head = None
        heatmap_Unet = None

    ho_model = model(cfg, backbone, neck, hand_sdf_head, obj_sdf_head, mano_head, volume_head, rot_head,awr_head,heatmap_Unet,heatmap_head)

    return ho_model


if __name__ == '__main__':
    model = get_model(cfg, True)
    input_size = (2, 3, 256, 256)
    input_img = torch.randn(input_size)
    input_point_size = (2, 2000, 3)
    input_points = torch.randn(input_point_size)
    sdf_results, hand_pose_results, obj_pose_results = model(input_img, input_points)