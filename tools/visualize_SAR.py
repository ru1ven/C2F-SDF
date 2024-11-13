import matplotlib
import cv2
import pyrender
import trimesh
import numpy as np

matplotlib.use('AGG')  # 或者PDF, SVG或PS

color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little


def draw_2d_skeleton(image, pose_uv):
    """
    :param image: H x W x 3
    :param pose_uv: 21 x 2
    wrist,
    thumb_mcp, thumb_pip, thumb_dip, thumb_tip
    index_mcp, index_pip, index_dip, index_tip,
    middle_mcp, middle_pip, middle_dip, middle_tip,
    ring_mcp, ring_pip, ring_dip, ring_tip,
    little_mcp, little_pip, little_dip, little_tip
    :return:
    """
    assert pose_uv.shape[0] == 21
    skeleton_overlay = image.copy()
    marker_sz = 6
    line_wd = 3
    root_ind = 0

    for joint_ind in range(pose_uv.shape[0]):
        joint = pose_uv[joint_ind, 0].astype('int32'), pose_uv[joint_ind, 1].astype('int32')
        cv2.circle(
            skeleton_overlay, joint,
            radius=marker_sz, color=color_hand_joints[joint_ind] * np.array(255), thickness=-1,
            lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            root_joint = pose_uv[root_ind, 0].astype('int32'), pose_uv[root_ind, 1].astype('int32')
            cv2.line(
                skeleton_overlay, root_joint, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
        else:
            joint_2 = pose_uv[joint_ind - 1, 0].astype('int32'), pose_uv[joint_ind - 1, 1].astype('int32')
            cv2.line(
                skeleton_overlay, joint_2, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

    return skeleton_overlay


def render_mesh_multi_views(img, mesh_h, face_h, mesh_o, face_o, cam_param):
    # mesh
    mesh_h[:, 1] = -mesh_h[:, 1]
    mesh_h[:, 2] = -mesh_h[:, 2]
    ##point_h = np.mean(mesh_h, axis=0)

    mesh_o[:, 1] = -mesh_o[:, 1]
    mesh_o[:, 2] = -mesh_o[:, 2]
    ##point_o = np.mean(mesh_o, axis=0)
    point_h =  np.mean(np.concatenate([mesh_o,mesh_h],0), axis=0)
    mesh_h = trimesh.Trimesh(mesh_h, face_h)
    mesh_o = trimesh.Trimesh(mesh_o, face_o)
    img_mesh = render_mesh(img, mesh_h, mesh_o, cam_param)
    white_img = np.zeros_like(img) + 255
    # white_img = np.zeros([10000, 10000, 1]) + 255
    mesh_view1h = mesh_h.copy()
    mesh_view1o = mesh_o.copy()
    rot = trimesh.transformations.rotation_matrix(  # rotate about axis defined by point anddirection.
        np.radians(-60), [0, 1, 0], point_h)  # 将角度从度数转换为弧度 180,1,1,0            60,0,1,0
    mesh_view1h.apply_transform(rot)
    mesh_view1o.apply_transform(rot)
    view_1 = render_mesh(white_img, mesh_view1h, mesh_view1o, cam_param)
    mesh_view2h = mesh_h.copy()
    mesh_view2o = mesh_o.copy()
    rot = trimesh.transformations.rotation_matrix(
        np.radians(60), [0, 1, 1], point_h)
    mesh_view2h.apply_transform(rot)
    mesh_view2o.apply_transform(rot)
    view_2 = render_mesh(white_img, mesh_view2h, mesh_view2o, cam_param)
    mesh_view3h = mesh_h.copy()
    mesh_view3o = mesh_o.copy()
    rot = trimesh.transformations.rotation_matrix(  # rotate about axis defined by point anddirection.
        np.radians(180), [0, 1, 0], point_h)  # 将角度从度数转换为弧度 180,1,1,0            60,0,1,0
    mesh_view3h.apply_transform(rot)
    mesh_view3o.apply_transform(rot)
    view_3 = render_mesh(white_img, mesh_view3h, mesh_view3o, cam_param)
    return img_mesh, view_1, view_2, view_3


def render_mesh(img, mesh_h, mesh_o, cam_intr):
    material_h = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
                                                    baseColorFactor=(0.6, 0.6, 0.8, 1))  # (B, G, R, ?)
    material_o = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE',
                                                    baseColorFactor=(0.8, 0.8, 0.6, 1))
    mesh_h = pyrender.Mesh.from_trimesh(mesh_h, material=material_h,
                                        smooth=False)  # Create a Mesh from a :class:`~trimesh.base.Trimesh, #The material of the object. Overrides any mesh material.
    # If `True`, the mesh will be rendered with interpolated vertex normals. Otherwise, the mesh edges will stay sharp.
    mesh_o = pyrender.Mesh.from_trimesh(mesh_o, material=material_o, smooth=False)
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))  # Color of ambient light
    scene.add(mesh_h, 'mesh_h')
    scene.add(mesh_o, 'mesh_o')

    # focal, princpt = cam_param['focal'], cam_param['princpt']
    fx = cam_intr[0, 0]
    fy = cam_intr[1, 1]
    cx = cam_intr[0, 2]
    cy = cam_intr[1, 2]
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy)
    scene.add(camera)  # nei can

    # renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=img.shape[1], viewport_height=img.shape[0],
                                          point_size=1.0)  # lixian xuanranqi

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0],
                                      intensity=0.8)  # Directional lights are light sources that act as though they are infinately far away and emit light in the direction of the local -z axis.
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])  #
    scene.add(light, pose=light_pose)  # The local pose of this node relative to its parent node.

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)  # xuanran tupian
    rgb = rgb[:, :, :3].astype(np.float32)
    # device = img.device()
    valid_mask = (depth > 0)[:, :, None]
    # save to image

    img = rgb * valid_mask + img * (1 - valid_mask)
    return img


if __name__ == '__main__':
    import numpy as np
