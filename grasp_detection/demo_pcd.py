import os
import sys
import argparse
import numpy as np
import open3d as o3d

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--pcd_path', required=True, help='Path to .pcd file')
parser.add_argument('--max_gripper_width', type=float, default=0.1)
parser.add_argument('--gripper_height', type=float, default=0.03)
parser.add_argument('--top_down_grasp', action='store_true')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--no_collision', action='store_true', help='Disable collision detection')
parser.add_argument('--dense_grasp', action='store_true', help='Enable dense grasp mode')
parser.add_argument('--scale', type=float, default=1000.0,
                    help='Scale factor to convert point cloud units to meters (default 1000 for mm->m)')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))


def demo():
    # Load PCD
    pcd_raw = o3d.io.read_point_cloud(cfgs.pcd_path)
    points = np.asarray(pcd_raw.points, dtype=np.float32) / cfgs.scale  # convert to meters
    colors = np.asarray(pcd_raw.colors, dtype=np.float32)               # already [0,1]

    print(f'Loaded {len(points)} points')
    print(f'XYZ min: {points.min(axis=0)}, max: {points.max(axis=0)}')

    # Build workspace lims from point cloud extents with a small margin
    xmin, xmax = float(points[:, 0].min()), float(points[:, 0].max())
    ymin, ymax = float(points[:, 1].min()), float(points[:, 1].max())
    zmin, zmax = float(points[:, 2].min()), float(points[:, 2].max())
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]
    print(f'Workspace lims (m): {lims}')

    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    gg, cloud = anygrasp.get_grasp(
        points, colors,
        lims=lims,
        apply_object_mask=True,
        dense_grasp=cfgs.dense_grasp,
        collision_detection=not cfgs.no_collision
    )

    if gg is None or len(gg) == 0:
        print('No grasps detected after collision detection!')
        if cfgs.debug:
            o3d.visualization.draw_geometries([cloud], window_name='Point Cloud (no grasps)')
        return

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:20]
    print(f'Top grasp scores: {gg_pick.scores}')
    print(f'Best grasp score: {gg_pick[0].score}')

    if cfgs.debug:
        trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        for g in grippers:
            g.transform(trans_mat)
        o3d.visualization.draw_geometries([*grippers, cloud], window_name='All grasps')
        o3d.visualization.draw_geometries([grippers[0], cloud], window_name='Best grasp')


if __name__ == '__main__':
    demo()
