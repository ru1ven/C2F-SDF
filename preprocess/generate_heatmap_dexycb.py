
import argparse
import yaml
import numpy as np
import torch
import os
import os.path as osp
from matplotlib import pyplot as plt
from tqdm import tqdm
from fire import Fire
from multiprocessing import Process, Queue
import trimesh
import sys
sys.path.insert(0, '../common')
sys.path.insert(0, '..')
from utils.camera import PerspectiveCamera
from mano.manolayer import ManoLayer
from datasets.dexycb.toolkit.factory import get_dataset

CUDA=False
voxel_heatmap_size = 32

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', '-e', default='../playground/experiments/DexYCB.yaml',type=str)
    parser.add_argument('--mode', '-m', default='train', type=str)
    parser.add_argument('--testset', '-testset', default='dexycb', type=str)
    parser.add_argument('--num_proc', default=1, type=int)
    parser.add_argument('--vis', action='store_true')
    args = parser.parse_args()
    return args


def generate_hm(dataset, db, mode='train', vis=False, data_root='../datasets/dexycb', input_image_size=256, cube=500,
                recon_scale=6.2):
    sdf_data_root = os.path.join(data_root, 'data')
    for idx, sample in tqdm(enumerate(db.data)):
        img_path = sample['img_path']

        img_info = dict()

        subject = int(img_path.split('/')[-4].split('-')[-1])
        video_id = img_path.split('/')[-3]
        sub_video_id = img_path.split('/')[-2]
        frame_idx = int(img_path.split('/')[-1].split('_')[-1].split('.')[0])

        fx = sample['fx']
        fy = sample['fy']
        cx = sample['cx']
        cy = sample['cy']
        bbox = sample['bbox']

        camera = PerspectiveCamera(fx, fy, cx, cy)
        camera.update_virtual_camera_after_crop(bbox)
        camera.update_intrinsics_after_resize((bbox[-1], bbox[-2]), (input_image_size, input_image_size))

        assert frame_idx % 5 == 0

        img_info['id'] = idx
        img_info['file_name'] = '_'.join([str(subject), video_id, sub_video_id, str(frame_idx)])

        assert (os.path.exists(
            os.path.join(sdf_data_root, 'norm', img_info['file_name'] + '.npz')) and mode == 'train') or mode == 'test'

        label = np.load(sample['seg_path'])

        mano_layer = ManoLayer(flat_hand_mean=False, ncomps=45, side='right',
                               mano_root='../common/mano/assets/', use_pca=True)
        betas = torch.tensor(sample['hand_shapes'], dtype=torch.float32).unsqueeze(0)
        hand_verts, _, hand_poses, _, _ = mano_layer(torch.from_numpy(label['pose_m'][:, 0:48]), betas,
                                                     torch.from_numpy(label['pose_m'][:, 48:51]))
        hand_verts = hand_verts[0].numpy().tolist()

        grasp_obj_id = sample['ycb_id']

        obj_rest_mesh = trimesh.load(dataset.obj_file[grasp_obj_id], process=False)
        obj_faces = obj_rest_mesh.faces

        meta_file = os.path.join(sdf_data_root,img_path.split('/')[-4], video_id, "meta.yml")
        with open(meta_file, 'r') as f:
            meta = yaml.load(f, Loader=yaml.FullLoader)
        ycb_grasp_ind = meta['ycb_grasp_ind']
        pose_y = label['pose_y'][ycb_grasp_ind]

        homo_obj_verts = np.ones((obj_rest_mesh.vertices.shape[0], 4))
        homo_obj_verts[:, :3] = obj_rest_mesh.vertices
        obj_verts = np.dot(pose_y, homo_obj_verts.transpose(1, 0)).transpose(1, 0)
        gt_obj_mesh = trimesh.Trimesh(vertices=obj_verts, faces=obj_faces)

        gt_hand_points = hand_verts
        gt_obj_points = obj_verts

        hand_hm_data_root = os.path.join(sdf_data_root, 'heatmap','heatmap_hand')
        obj_hm_data_root = os.path.join(sdf_data_root, 'heatmap','heatmap_obj')

        os.makedirs(hand_hm_data_root, exist_ok=True)
        os.makedirs(obj_hm_data_root, exist_ok=True)
        gt_hand_points = torch.Tensor(gt_hand_points)
        rot_cam_extr = torch.from_numpy(camera.extrinsics[:3, :3].T)
        hand_joints_3d = torch.from_numpy(sample['hand_joints_3d'])

        gt_obj_points = torch.Tensor(gt_obj_points)
        if CUDA is True:
            gt_hand_points = gt_hand_points.cuda()
            gt_obj_points = gt_obj_points.cuda()
            rot_cam_extr = rot_cam_extr.cuda()
            hand_joints_3d=hand_joints_3d.cuda()

        gt_hand_points[:, 0:3] = torch.mm(rot_cam_extr, gt_hand_points[:, 0:3].transpose(1, 0)).transpose(1, 0)
        gt_obj_points[:, 0:3] = torch.mm(rot_cam_extr, gt_obj_points[:, 0:3].transpose(1, 0)).transpose(1, 0)

        hand_joints_3d[:, 0:3] = torch.mm(rot_cam_extr, hand_joints_3d[:, 0:3].transpose(1, 0)).transpose(1, 0)

        gt_hand_xyz_normal = (gt_hand_points - hand_joints_3d[0]) * recon_scale / 2
        gt_obj_xyz_normal = (gt_obj_points - hand_joints_3d[0]) * recon_scale / 2

        hm_hand = verts2hm_xyz_gs(gt_hand_xyz_normal, feature_size=voxel_heatmap_size).cpu()
        hm_obj = verts2hm_xyz_gs(gt_obj_xyz_normal, feature_size=voxel_heatmap_size).cpu()
        if not vis:
            npz_filename_out = osp.join(hand_hm_data_root, img_info['file_name'] + '.npy')
            np.save(npz_filename_out, hm_hand)
            npz_filename_out_obj = osp.join(obj_hm_data_root, img_info['file_name'] + '.npy')
            np.save(npz_filename_out_obj, hm_obj)
        # else:
        #     vis_hm(hm_hand)
        #     vis_hm(hm_obj)


def vis_hm(hm):
    for idx in range(voxel_heatmap_size):
        data = np.array(hm[:, :, idx])
        plt.imshow(data)
        plt.axis('off')
        plt.clim(0, 1)
        # plt.colorbar()
        plt.show()


def main():
    # argument parse and create log
    args = parse_args()
    # testset = args.dir.strip('/').split('/')[-1].split('_')[1]
    testset = args.testset
    exec(f'from datasets.{testset}.{testset} import {testset}')

    num_samples = 29656 if args.mode == 'train' else 5928

    start_points = []
    end_points = []

    dataset_name = f's0_{args.mode}'
    dataset = get_dataset(dataset_name)

    division = num_samples // args.num_proc

    for i in range(args.num_proc):
        start_point = i * division
        if i != args.num_proc - 1:
            end_point = start_point + division
        else:
            end_point = num_samples
        start_points.append(start_point)
        end_points.append(end_point)

    process_list = []
    for i in range(args.num_proc):
        if args.mode == 'train':
            testset_db = eval(testset)(args.mode + '_s0_29k', start_points[i], end_points[i])
        else:
            testset_db = eval(testset)(args.mode + '_s0_5k', start_points[i], end_points[i])
        p = Process(target=generate_hm, args=(dataset, testset_db, args.mode, args.vis))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()


def verts2hm(verts, feature_size, kernel_size=0.8):
    device = verts.device

    verts_num, _ = verts.view(-1, 3).size()
    verts_feature = verts.reshape(verts.size(0), -1, 1, 1, 1).repeat(1, 1, feature_size, feature_size, feature_size)
    mesh_x = 2.0 * (torch.arange(feature_size).unsqueeze(1).unsqueeze(1).expand(feature_size, feature_size,
                                                                                feature_size).float() + 0.5) / feature_size - 1.0
    mesh_y = 2.0 * (torch.arange(feature_size).unsqueeze(0).unsqueeze(1).expand(feature_size, feature_size,
                                                                                feature_size).float() + 0.5) / feature_size - 1.0
    mesh_z = 2.0 * (torch.arange(feature_size).unsqueeze(0).unsqueeze(0).expand(feature_size, feature_size,
                                                                                feature_size).float() + 0.5) / feature_size - 1.0
    coords = torch.stack((mesh_y, mesh_x, mesh_z), dim=0)
    coords = coords.repeat(verts_num, 1, 1, 1, 1)

    offset = verts_feature - coords
    offset = offset.view(verts_num, 3, feature_size, feature_size, feature_size)
    dist = torch.sqrt(torch.sum(torch.pow(offset, 2), dim=1) + 1e-8)

    if torch.is_tensor(kernel_size):
        kernel_size = kernel_size.to(device)
        heatmap = (kernel_size.view(verts_num, 1, 1, 1) - dist) / kernel_size.view(verts_num, 1, 1, 1)
    else:
        heatmap = (kernel_size - dist) / kernel_size
    mask = heatmap.ge(0).float().view(verts_num, feature_size, feature_size, feature_size)
    heatmap_mask = heatmap * mask.float()

    heatmap_mask = torch.sum(heatmap_mask, dim=0)
    heatmap_mask /= heatmap_mask.max()
    return heatmap_mask


def verts2hm_xyz_gs(verts, feature_size, std= 0.3, sigma = 2, threshold = 10):
    device = verts.device

    verts_num, _ = verts.view(-1, 3).size()
    verts[:, 0] = (verts[:, 0] + 1) / 2 * feature_size
    verts[:, 1] = (verts[:, 1] + 1) / 2 * feature_size
    verts[:, 2] = (verts[:, 2] + 1) / 2 * feature_size

    verts_x = verts[:, 0].view(verts_num, 1, 1, 1).repeat(1, feature_size, feature_size, feature_size).float()
    verts_y = verts[:, 1].view(verts_num, 1, 1, 1).repeat(1, feature_size, feature_size, feature_size).float()
    verts_z = verts[:, 2].view(verts_num, 1, 1, 1).repeat(1, feature_size, feature_size, feature_size).float()

    xx, yy, zz = np.meshgrid(np.arange(feature_size), np.arange(feature_size), np.arange(feature_size))

    mesh_x = torch.from_numpy(xx).view(1, feature_size, feature_size, feature_size).repeat(verts_num, 1, 1, 1).float().to(device)
    mesh_y = torch.from_numpy(yy).view(1, feature_size, feature_size, feature_size).repeat(verts_num, 1, 1, 1).float().to(device)
    mesh_z = torch.from_numpy(zz).view(1, feature_size, feature_size, feature_size).repeat(verts_num, 1, 1, 1).float().to(device)

    heatmap = torch.exp(
        -(torch.pow((mesh_y - verts_x) / std, 2) + torch.pow((mesh_x - verts_y) / std, 2) +torch.pow((mesh_z - verts_z) / std, 2)) / (2 * pow(sigma, 2)))


    mask = heatmap.ge(0).float().view(verts_num, feature_size, feature_size, feature_size)
    heatmap_mask = heatmap * mask.float()


    heatmap_mask = torch.sum(heatmap_mask, dim=0)
    heatmap_mask[heatmap_mask >= threshold] = threshold

    heatmap_mask /= (heatmap_mask.max()+1e-8)

    return heatmap_mask


if __name__ == '__main__':
    Fire(main)
    # Fire(create_lmdb)
