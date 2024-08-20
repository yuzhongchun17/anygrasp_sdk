import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

def demo(data_dir):
    # print('Debug')
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # get data
    colors = np.array(Image.open(os.path.join(data_dir, 'image.png')), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(data_dir, 'depth_hl_1.png'))) 
    # print(depths)
    # print(f'color shape:{colors.shape} and depth shape:{depths.shape}')
    
    # # get camera intrinsics
    # fx, fy = 927.17, 927.37
    # cx, cy = 651.32, 349.62
    # scale = 1000.0
    # fx, fy, cx, cy, scale = 306, 306, 118, 211, 1000
    K = [1481.8690580024659, 0.0, 936.7853966108089, 0.0, 1485.0033867902052, 511.25986612371463, 0.0, 0.0, 1.0] # pv camera instrinsics (3x3)
    fx, fy, cx, cy, scale = K[0], K[4], K[2], K[5], 1000
    # set workspace to filter output grasps
    # xmin, xmax = -0.19, 0.12
    # ymin, ymax = 0.02, 0.15
    # zmin, zmax = 0.0, 1.0
    # lims = [xmin, xmax, ymin, ymax, zmin, zmax]
    # print(f'lims:{lims}')

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths / scale
    # print(points_z)
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    mask = (points_z > 0.3) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    # print(f'points array size:{points.shape}')
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    # print(f'colors array size:{colors.shape}')
    print(points.min(axis=0), points.max(axis=0))

    # # gg is a list of grasps of type graspgroup in graspnetAPI
    xmin = points_x.min()
    xmax = points_x.max()
    ymin = points_y.min()
    ymax = points_y.max()
    zmin = 0.4
    zmax = 0.8
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]
    # print(f'lims2:{lims2}')

    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:20]
    print(gg_pick.scores)
    print('grasp score:', gg_pick[0].score)
    print('grasp pose:', gg_pick[0])

    # visualization
    if cfgs.debug:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        o3d.visualization.draw_geometries([*grippers, cloud])
        o3d.visualization.draw_geometries([grippers[0], cloud])


if __name__ == '__main__':
    
    demo('./example_data/')
