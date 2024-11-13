#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import sys
from yacs.config import CfgNode as CN
from loguru import logger
from contextlib import redirect_stdout


cfg = CN()

cfg.add_info = 'RGBDSDF+interpolation+classifyPCL'

cfg.task = 'sdf_pcl_interpolation_classifyPCL'
cfg.cur_dir = osp.dirname(os.path.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, '', '..')
cfg.data_dir = osp.join(cfg.root_dir, 'datasets')
cfg.output_dir = 'sdf_pcl_interpolation_classifyPCL'
cfg.model_dir = './model_dump'
cfg.vis_dir = './vis'
cfg.log_dir = './log'
cfg.result_dir = './result'
cfg.sdf_result_dir = 'sdf_pcl_interpolation_classifyPCL'
cfg.cls_sdf_result_dir = 'sdf_pcl_interpolation_classifyPCL'
cfg.hand_pose_result_dir = 'sdf_pcl_interpolation_classifyPCL'
cfg.optim_hand_pose_result_dir = 'sdf_pcl_interpolation_classifyPCL'
cfg.obj_pose_result_dir = 'sdf_pcl_interpolation_classifyPCL'
cfg.with_add_feats = True
cfg.obj_hm_sup = False
cfg.K=16

## dataset
cfg.trainset_3d = 'dexycb'
cfg.trainset_3d_split = 's0_29k'
cfg.testset = 'dexycb'
cfg.testset_split = 's0_5k'
cfg.testset_hand_source = osp.join(cfg.testset, 'data/test/mesh_hand')
cfg.testset_obj_source = osp.join(cfg.testset, 'data/test/mesh_obj')
cfg.num_testset_samples = 6285
cfg.mesh_resolution = 128
cfg.point_batch_size = 2 ** 18
cfg.output_part_label = False
cfg.vis_part_label = False
cfg.chamfer_optim = True

## model setting
cfg.modal = 'RGB'
cfg.backbone = 'resnet_18'
cfg.mano_pca_latent = 15
cfg.sdf_latent = 256
cfg.hand_point_latent = 3
cfg.obj_point_latent = 3
cfg.hand_encode_style = 'nerf'
cfg.obj_encode_style = 'nerf'
cfg.rot_format = 'axisang'
cfg.hand_branch = True
cfg.obj_branch = True
cfg.hand_cls = False
cfg.mano_branch = False
cfg.mano_depth = False
cfg.obj_trans = False
cfg.obj_rot = False
cfg.awr_branch = False
cfg.pcl_branch = True
cfg.classifyPCL = True
cfg.sdf_weight= True

cfg.sdf_head = CN()
cfg.sdf_head.layers = 5
cfg.sdf_head.dims = [512 for i in range(cfg.sdf_head.layers - 1)]
cfg.sdf_head.dropout = [i for i in range(cfg.sdf_head.layers - 1)]
cfg.sdf_head.norm_layers = [i for i in range(cfg.sdf_head.layers - 1)]
cfg.sdf_head.dropout_prob = 0.2
cfg.sdf_head.latent_in = [(cfg.sdf_head.layers - 1) // 2]
cfg.sdf_head.num_class = 6
cfg.pcl_sample_nums = 1024

## training config
cfg.image_size = (256, 256)
cfg.heatmap_size = (64, 64, 64)
cfg.voxel_heatmap_size = 32
cfg.hm_threshold = 0.005
cfg.depth_dim = 0.28
cfg.warm_up_epoch = 0
cfg.lr_dec_epoch = [600, 1200]
cfg.end_epoch = 800
#1600
cfg.sdf_add_epoch = 1201
cfg.pose_epoch = 0
cfg.lr = 1e-4
cfg.lr_dec_style = 'step'
cfg.lr_dec_factor = 0.5
cfg.train_batch_size = 128
cfg.num_sample_points = 2000
cfg.clamp_dist = 0.05
cfg.recon_scale = 6.5
cfg.hand_sdf_weight = 0.5
cfg.obj_sdf_weight = 0.5
cfg.hand_cls_weight = 0.05
cfg.mano_joints_weight = 0.5
cfg.pixel_weight = 0.001
cfg.heatmap_weight = 0.01
cfg.coord_weight = 0.1
cfg.mano_shape_weight = 5e-7
cfg.mano_pose_weight = 5e-5
cfg.volume_weight = 0.5
cfg.corner_weight = 0.2
cfg.use_inria_aug = False
cfg.number_joints = 21

## testing config
cfg.test_batch_size = 1
cfg.test_with_gt = False
cfg.heatmap = False

## others
cfg.use_lmdb = True
cfg.num_threads = 8
cfg.gpu_ids = (0, 1, 2, 3)
cfg.num_gpus = 4
cfg.checkpoint = 'model.pth.tar'
cfg.model_save_freq = 400

def update_config(cfg, args, mode='train'):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.gpu_ids = args.gpu_ids
    #cfg.model_dir = args.model_dir
    cfg.num_gpus = len(cfg.gpu_ids.split(','))
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_ids
    logger.info('>>> Using GPU: {}'.format(cfg.gpu_ids))

    if mode == 'train':

        exp_info = [cfg.trainset_3d + cfg.trainset_3d_split, cfg.backbone.replace('_', ''), 'h' + str(int(cfg.hand_branch)), 'o' + str(int(cfg.obj_branch)), 'sdf' + str(cfg.sdf_head.layers), 'cls' + str(int(cfg.hand_cls)), 'mano' + str(int(cfg.mano_branch)), 'd' + str(int(cfg.mano_depth)), 'ot' + str(int(cfg.obj_trans)), 'or' + str(int(cfg.obj_rot)), 'hand_' + cfg.hand_encode_style + '_' + str(cfg.hand_point_latent), 'obj_' + cfg.obj_encode_style + '_' + str(cfg.obj_point_latent), 'np' + str(cfg.num_sample_points), 'e' + str(cfg.end_epoch), 'ae' + str(cfg.sdf_add_epoch), 'pe' + str(cfg.pose_epoch), 'scale' + str(cfg.recon_scale), 'b' + str(cfg.num_gpus * cfg.train_batch_size), 'hsw' + str(cfg.hand_sdf_weight), 'osw' + str(cfg.obj_sdf_weight), 'hcw' + str(cfg.hand_cls_weight), 'mjw' + str(cfg.mano_joints_weight), 'msw' + str(cfg.mano_shape_weight), 'mpw' + str(cfg.mano_pose_weight), 'vw' + str(cfg.volume_weight), 'ocrw' + str(cfg.corner_weight)]

        cfg.output_dir = osp.join(cfg.root_dir, 'outputs', cfg.task, cfg.add_info,'_'.join(exp_info))
        cfg.model_dir = osp.join(cfg.output_dir, 'model_dump')
        cfg.vis_dir = osp.join(cfg.output_dir, 'vis')
        cfg.log_dir = osp.join(cfg.output_dir, 'log')
        cfg.result_dir = osp.join(cfg.output_dir, '_'.join(['result', cfg.testset, 'gt', str(int(cfg.test_with_gt))]))
        cfg.sdf_result_dir = osp.join(cfg.result_dir, 'sdf_mesh')
        cfg.cls_sdf_result_dir = osp.join(cfg.result_dir, 'hand_cls')
        cfg.hand_pose_result_dir = os.path.join(cfg.result_dir, 'hand_pose')
        cfg.optim_hand_pose_result_dir = os.path.join(cfg.result_dir, 'optim_hand_pose')
        cfg.obj_pose_result_dir = os.path.join(cfg.result_dir, 'obj_pose')

        os.makedirs(cfg.output_dir, exist_ok=True)
        os.makedirs(cfg.model_dir, exist_ok=True)
        os.makedirs(cfg.vis_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.result_dir, exist_ok=True)
        os.makedirs(cfg.sdf_result_dir, exist_ok=True)
        os.makedirs(cfg.cls_sdf_result_dir, exist_ok=True)
        os.makedirs(cfg.hand_pose_result_dir, exist_ok=True)
        os.makedirs(cfg.optim_hand_pose_result_dir, exist_ok=True)
        os.makedirs(cfg.obj_pose_result_dir, exist_ok=True)

        cfg.freeze()
        with open(osp.join(cfg.output_dir, 'exp.yaml'), 'w') as f:
            with redirect_stdout(f): print(cfg.dump())
    else:
        cfg.model_dir = args.model_dir
        #cfg.model_dir = osp.join(cfg.output_dir, 'model_dump')

        cfg.result_dir = osp.join(cfg.root_dir, 'outputs', cfg.task, cfg.add_info)
        cfg.log_dir = osp.join(cfg.result_dir, 'test_log')
        cfg.sdf_result_dir = osp.join(cfg.result_dir, 'sdf_mesh')
        cfg.cls_sdf_result_dir = osp.join(cfg.result_dir, 'hand_cls')
        cfg.hand_pose_result_dir = os.path.join(cfg.result_dir, 'hand_pose')
        cfg.optim_hand_pose_result_dir = os.path.join(cfg.result_dir, 'optim_hand_pose')
        cfg.obj_pose_result_dir = os.path.join(cfg.result_dir, 'obj_pose')
        os.makedirs(cfg.result_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.sdf_result_dir, exist_ok=True)
        os.makedirs(cfg.cls_sdf_result_dir, exist_ok=True)
        os.makedirs(cfg.hand_pose_result_dir, exist_ok=True)
        os.makedirs(cfg.optim_hand_pose_result_dir, exist_ok=True)
        os.makedirs(cfg.obj_pose_result_dir, exist_ok=True)

        cfg.freeze()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir_utils import add_pypath
add_pypath(osp.join(cfg.data_dir))
add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d))
add_pypath(osp.join(cfg.data_dir, cfg.testset))