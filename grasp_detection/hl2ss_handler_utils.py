import cv2
import numpy as np
from PIL import Image
import open3d as o3d

def get_bbox(mask) -> tuple:
    """
    Get the bounding box of a binary mask. 
    bbox -> [x_min, x_max, y_min, y_max]
    """
    x, y, w, h = cv2.boundingRect(mask)
    return x, x+w, y, y+h


def project_3d_to_2d(point, camera_info):
    """
    Project a 3D point to a 2D pixel. [X, Y, Z] -> [x, y]
    """
    X, Y, Z = point
    fx, fy, cx, cy = camera_info.K[0], camera_info.K[4], camera_info.K[2], camera_info.K[5]
    image_x = (X/Z) * fx + cx
    image_y = (Y/Z) * fy + cy
    return int(image_x), int(image_y)

def get_cam_ee_rotation() -> np.ndarray:
    """
    Get the rotation matrix of the camera in the robot end effector frame.
    """
    Rot_cam_ee = np.array([[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]])
    return Rot_cam_ee

def visualize_pcd(rgb_image, depth_image, camera_info):
    """
    Visualize a point cloud from RGB and depth images.
    """
    # Create Open3D color and depth images
    color_image = o3d.geometry.Image(rgb_image.astype(np.uint8))
    depth_image = o3d.geometry.Image(depth_image.astype(np.float32))

    # Create RGBD image from color and depth images
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,
        depth_scale=1.0,  # Assume depth scale is 1000 for converting to meters
        depth_trunc=2.0,     # Truncate depth beyond __ [m]
        convert_rgb_to_intensity=False
    )

    # Create a point cloud from the RGBD image
    fx, fy, cx, cy, scale = camera_info.K[0], camera_info.K[4], camera_info.K[2], camera_info.K[5], 1
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            camera_info.width,  # Assume intrinsic parameters are stored in cam_K
            camera_info.height,
            fx,
            fy,
            cx,
            cy
        )
    )

    # Initialize visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add point cloud to visualizer
    vis.add_geometry(pcd)
    
    # Run the visualizer
    vis.run()
    # vis.destroy_window()
