
import argparse

import numpy as np
import torch
import os
import os.path as osp
from matplotlib import pyplot as plt
from tqdm import tqdm
from fire import Fire
from multiprocessing import Process
import pickle
import sys

from preprocess.generate_heatmap_dexycb import verts2hm_xyz_gs

sys.path.insert(0, '../common')
sys.path.insert(0, '..')

CUDA=False
voxel_heatmap_size = 32

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cfg', '-e', default='../playground/experiments/DexYCB.yaml',type=str)
    parser.add_argument('--mode', '-m', default='train', type=str)
    parser.add_argument('--testset', '-testset', default='obman', type=str)
    parser.add_argument('--num_proc', default=1, type=int)
    parser.add_argument('--vis', action='store_true')
    args = parser.parse_args()
    return args


def generate_hm(db, mode='train', vis=False, data_root='../datasets/obman', input_image_size=256, cube=500,
                recon_scale=7.0):
    sdf_data_root = os.path.join(data_root, 'data')
    data_path = osp.join(data_root, 'data', mode)
    meta_path_gSDF = osp.join(data_path, 'meta_gSDF')
    meta_path = osp.join(data_path, 'meta')
    cam_extr = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float32)
    obj_path = os.path.join("/data/data3/pfren/dataset/object/ShapeNetCore.v2", "{}/{}/models/model_normalized.obj")
    for idx, sample in tqdm(enumerate(db.data)):
        # if idx > 0:
        #     break

        img_path = sample['img_path']
        img_info = dict()

        filename = sample['id']
        meta_data_file_gSDF = osp.join(meta_path_gSDF, filename+'.pkl')
        with open(meta_data_file_gSDF, 'rb') as f:
            meta_info_gSDF = pickle.load(f)
            print(meta_info_gSDF)

        meta_data_file = osp.join(meta_path, filename + '.pkl')
        with open(meta_data_file, 'rb') as f:
            meta_info = pickle.load(f)
            print(meta_info)


        hand_verts = np.dot(cam_extr, meta_info['verts_3d'].transpose(1, 0)).transpose(1, 0).tolist()
        obj_path = obj_path.format(meta_info["class_id"], meta_info["sample_id"])
        print(obj_path)

        obj_trans = meta_info["affine_transform"]
        with open(obj_path, 'r') as m_f:
            mesh = fast_load_obj(m_f)[0]
        verts = mesh['vertices']
        # Apply transforms
        hom_verts = np.concatenate([verts, np.ones([verts.shape[0], 1])], axis=1)
        trans_verts = obj_trans.dot(hom_verts.T).T[:, :3]
        trans_verts = cam_extr[:3, :3].dot(
            trans_verts.transpose()).transpose()
        trans_verts, ofaces = np.array(trans_verts).astype(np.float32), np.array(mesh['faces']).astype(np.int16)
        #trans_verts = trans_verts * 1000

        if idx <= 10:
            np.savetxt('/home/pfren/lxy/visual/v_hand_{}.txt'.format(filename), hand_verts, fmt='%.3f')
            np.savetxt('/home/pfren/lxy/visual/v_obj_{}.txt'.format(filename), trans_verts, fmt='%.3f')

        # gt_hand_points, _ = trimesh.sample.sample_surface(gt_hand_mesh, 500)
        gt_hand_points = hand_verts
        gt_obj_points = trans_verts

        hand_hm_data_root = os.path.join(sdf_data_root, mode, 'heatmap', 'heatmap_hand')
        obj_hm_data_root = os.path.join(sdf_data_root, mode,'heatmap','heatmap_obj')


        os.makedirs(hand_hm_data_root, exist_ok=True)
        os.makedirs(obj_hm_data_root, exist_ok=True)
        gt_hand_points = torch.Tensor(gt_hand_points)
        rot_cam_extr = torch.from_numpy(cam_extr[:3, :3].T)
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

        hm_hand = verts2hm_xyz_gs(gt_hand_xyz_normal, feature_size=voxel_heatmap_size, std= 0.2, sigma = 3, max_therod=5).cpu()
        hm_obj = verts2hm_xyz_gs(gt_obj_xyz_normal, feature_size=voxel_heatmap_size, std= 0.2, sigma = 3, max_therod=5).cpu()
        if not vis:
            npz_filename_out = osp.join(hand_hm_data_root, filename + '.npy')
            np.save(npz_filename_out, hm_hand)
            npz_filename_out_obj = osp.join(obj_hm_data_root,filename + '.npy')
            np.save(npz_filename_out_obj, hm_obj)
        # else:
        #     vis_hm(hm_hand)
        #     vis_hm(hm_obj)


def main():
    # argument parse and create log
    args = parse_args()
    # testset = args.dir.strip('/').split('/')[-1].split('_')[1]
    testset = args.testset
    exec(f'from datasets.{testset}.{testset} import {testset}')

    num_samples = 87190 if args.mode == 'train' else 6285

    start_points = []
    end_points = []

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
            testset_db = eval(testset)(args.mode + '_87k', start_points[i], end_points[i])
        else:
            testset_db = eval(testset)(args.mode + '_6k', start_points[i], end_points[i])
        p = Process(target=generate_hm, args=(testset_db, args.mode, args.vis))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()






def fast_load_obj(file_obj, **kwargs):
    """
    Code slightly adapted from trimesh (https://github.com/mikedh/trimesh)
    Thanks to Michael Dawson-Haggerty for this great library !
    loads an ascii wavefront obj file_obj into kwargs
    for the trimesh constructor.

    vertices with the same position but different normals or uvs
    are split into multiple vertices.

    colors are discarded.

    parameters
    ----------
    file_obj : file object
                   containing a wavefront file

    returns
    ----------
    loaded : dict
                kwargs for trimesh constructor
    """

    # make sure text is utf-8 with only \n newlines
    text = file_obj.read()
    if hasattr(text, 'decode'):
        text = text.decode('utf-8')
    text = text.replace('\r\n', '\n').replace('\r', '\n') + ' \n'

    meshes = []

    def append_mesh():
        # append kwargs for a trimesh constructor
        # to our list of meshes
        if len(current['f']) > 0:
            # get vertices as clean numpy array
            vertices = np.array(
                current['v'], dtype=np.float64).reshape((-1, 3))
            # do the same for faces
            faces = np.array(current['f'], dtype=np.int64).reshape((-1, 3))

            # get keys and values of remap as numpy arrays
            # we are going to try to preserve the order as
            # much as possible by sorting by remap key
            keys, values = (np.array(list(remap.keys())),
                            np.array(list(remap.values())))
            # new order of vertices
            vert_order = values[keys.argsort()]
            # we need to mask to preserve index relationship
            # between faces and vertices
            face_order = np.zeros(len(vertices), dtype=np.int64)
            face_order[vert_order] = np.arange(len(vertices), dtype=np.int64)

            # apply the ordering and put into kwarg dict
            loaded = {
                'vertices': vertices[vert_order],
                'faces': face_order[faces],
                'metadata': {}
            }

            # build face groups information
            # faces didn't move around so we don't have to reindex
            if len(current['g']) > 0:
                face_groups = np.zeros(len(current['f']) // 3, dtype=np.int64)
                for idx, start_f in current['g']:
                    face_groups[start_f:] = idx
                loaded['metadata']['face_groups'] = face_groups

            # we're done, append the loaded mesh kwarg dict
            meshes.append(loaded)

    attribs = {k: [] for k in ['v']}
    current = {k: [] for k in ['v', 'f', 'g']}
    # remap vertex indexes {str key: int index}
    remap = {}
    next_idx = 0
    group_idx = 0

    for line in text.split("\n"):
        line_split = line.strip().split()
        if len(line_split) < 2:
            continue
        if line_split[0] in attribs:
            # v, vt, or vn
            # vertex, vertex texture, or vertex normal
            # only parse 3 values, ignore colors
            attribs[line_split[0]].append([float(x) for x in line_split[1:4]])
        elif line_split[0] == 'f':
            # a face
            ft = line_split[1:]
            if len(ft) == 4:
                # hasty triangulation of quad
                ft = [ft[0], ft[1], ft[2], ft[2], ft[3], ft[0]]
            for f in ft:
                # loop through each vertex reference of a face
                # we are reshaping later into (n,3)
                if f not in remap:
                    remap[f] = next_idx
                    next_idx += 1
                    # faces are "vertex index"/"vertex texture"/"vertex normal"
                    # you are allowed to leave a value blank, which .split
                    # will handle by nicely maintaining the index
                    f_split = f.split('/')
                    current['v'].append(attribs['v'][int(f_split[0]) - 1])
                current['f'].append(remap[f])
        elif line_split[0] == 'o':
            # defining a new object
            append_mesh()
            # reset current to empty lists
            current = {k: [] for k in current.keys()}
            remap = {}
            next_idx = 0
            group_idx = 0

        elif line_split[0] == 'g':
            # defining a new group
            group_idx += 1
            current['g'].append((group_idx, len(current['f']) // 3))

    if next_idx > 0:
        append_mesh()

    return meshes


if __name__ == '__main__':
    Fire(main)
    # Fire(create_lmdb)
