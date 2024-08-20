import os
import numpy as np
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped

from tf2_ros import TransformBroadcaster, TransformListener
import tf2_ros
import message_filters


from cv_bridge import CvBridge

import argparse
import torch
import numpy as np
import open3d as o3d

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
from hl2ss_ros_utils import prepare_tf_msg

# Parse terminal input
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
    parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
    parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()


class ImageDataHandler:
    def __init__(self, cfgs):
        self.cfgs = cfgs

        self.depth_image = None
        self.rgb_image = None
        self.camera_info = None
        self.is_grasp_predicted = False
        
        self.target_gg = None
        self.tf_lt_tmp = TransformStamped() # the moment grasp is predicted
    
        # ROS ----------------------------------------------------------
        rospy.init_node('camera_data_handler', anonymous=True)
        # Initialize the CvBridge and TF2 listener
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Set up subscribers using message_filters 
        self.rgb_sub = message_filters.Subscriber('/hololens2/image_pv_remap', Image)
        self.depth_sub = message_filters.Subscriber('/hololens2/image_lt', Image)
        self.info_sub = message_filters.Subscriber('/hololens2/camerainfo_lt', CameraInfo)

        # Set up a TimeSynchronizer (for sync sub)
        self.ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub, self.info_sub], 10)
        self.ts.registerCallback(self.callback)

        self.br = TransformBroadcaster()
        # Timer to periodically try processing the images
        self.timer = rospy.Timer(rospy.Duration(0.5), self.timer_callback)  # Adjust duration as needed


    def callback(self, rgb_msg, depth_msg, info_msg):
        try:
            # Process images
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8") # decode
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            self.camera_info = info_msg
            # rospy.loginfo("Synchronized messages and TF received")
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"Error getting TF transform: {str(e)}")
        except Exception as e:
            rospy.logerr(f"Error processing synchronized messages: {str(e)}")


    def timer_callback(self, event):
        if self.check_data_ready() and not self.is_grasp_predicted:
            print("here")
            # self.debug_info()
            # self.visualize_pcd()
            # self.timestamp = self.camera_info.header.stamp
            self.predict_grasp()
            self.is_grasp_predicted = True
        if self.is_grasp_predicted:
            # print("br grasp_pose")
            self.br_target_pose()

    def check_data_ready(self):
        return self.depth_image is not None and self.rgb_image is not None and self.camera_info is not None

    # def debug_info(self):
    #     rospy.loginfo("Processing images...")
    #     print(f'Depth image shape: {self.depth_image.shape}')
    #     print(f'Color image shape: {self.rgb_image.shape}')
    #     print(f'Camera intrinsics K matrix: {self.camera_info.K}')

    def visualize_pcd(self):
        # Create Open3D color and depth images
        color_image = o3d.geometry.Image(self.rgb_image.astype(np.uint8))
        depth_image = o3d.geometry.Image(self.depth_image.astype(np.float32))

        # Create RGBD image from color and depth images
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_image,
            depth_image,
            depth_scale=1.0,  # Assume depth scale is 1000 for converting to meters
            depth_trunc=2.0,     # Truncate depth beyond __ [m]
            convert_rgb_to_intensity=False
        )

        # Create a point cloud from the RGBD image
        fx, fy, cx, cy, scale = self.camera_info.K[0], self.camera_info.K[4], self.camera_info.K[2], self.camera_info.K[5], 1

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                self.camera_info.width,  # Assume intrinsic parameters are stored in cam_K
                self.camera_info.height,
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

    def predict_grasp(self):
        anygrasp = AnyGrasp(cfgs)
        anygrasp.load_net()
       
        # set the time stamp when prediction happend
        self.t = self.camera_info.header.stamp
        # get the tf (/hl_world /lt), rgb, at time t
        self.tf_lt_tmp = self.tf_buffer.lookup_transform('hl_world', 'lt', self.t, rospy.Duration(1.0))

        fx, fy, cx, cy, scale = self.camera_info.K[0], self.camera_info.K[4], self.camera_info.K[2], self.camera_info.K[5], 1
        # print(f'time stamp for prediction:{self.camera_info.header.fx}')
        # get point cloud
        xmap, ymap = np.arange(self.depth_image.shape[1]), np.arange(self.depth_image.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = self.depth_image / scale
        # print(points_z)
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        # set your workspace to crop point cloud
        mask = (points_z > 0.3) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        # print(f'points array size:{points.shape}')
        points = points[mask].astype(np.float32)
        colors = self.rgb_image[mask].astype(np.float32) / 255.0 # change to [0,1] from [0,255]
        # print(f'colors array size:{colors.shape}')
        print(points.min(axis=0), points.max(axis=0))

        # gg is a list of grasps of type graspgroup in graspnetAPI
        xmin = points_x.min()
        xmax = points_x.max()
        ymin = points_y.min()
        ymax = points_y.max()
        zmin = 0.4
        zmax = 0.8
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)
        if len(gg) == 0:
            print('No Grasp detected after collision detection!')

        gg = gg.nms().sort_by_score()
        gg_pick = gg[0:20]

        best_gg = gg_pick[0]
        # print(gg_pick.scores)
        print('grasp score:', best_gg.score)
        # print(f'grasp translation:{type(best_gg.translation)} : and grasp rotation{type(best_gg.rotation_matrix)}')
        self.target_gg = best_gg

        # visualization
        if cfgs.debug:
            trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            cloud.transform(trans_mat)
            grippers = gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
            o3d.visualization.draw_geometries([*grippers, cloud])
            o3d.visualization.draw_geometries([grippers[0], cloud])
    
    def br_target_pose(self):
        # prepare tf for grasp pose (in lt frame)
        T_grasp_to_ee = np.array([[0,0,1],
                                  [1,0,0],
                                  [0,1,0]])
        rot_in_ee_frame = self.target_gg.rotation_matrix @ T_grasp_to_ee
        # tf_grasp_pose = prepare_tf_msg('lt_tmp', 'grasp_pose', 
        #                                None, 
        #                                self.target_gg.translation, 
        #                                self.target_gg.rotation_matrix)
        tf_grasp_pose = prepare_tf_msg('lt_tmp', 'grasp_pose', 
                                None, 
                                self.target_gg.translation, 
                                rot_in_ee_frame)
        # preprare tf for lt at time t (in hl_world frame)
        self.tf_lt_tmp.header.stamp = tf_grasp_pose.header.stamp
        self.tf_lt_tmp.child_frame_id = 'lt_tmp'

        self.br.sendTransform([tf_grasp_pose, self.tf_lt_tmp])

if __name__ == '__main__':
    cfgs = parse_arguments()
    cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

    handler = ImageDataHandler(cfgs)
    rospy.spin()



    # rospy.init_node('camera_data_handler', anonymous=True)
    # self.depth_sub = rospy.Subscriber("/hololens2/image_lt", Image, self.depth_callback)
    # self.rgb_sub = rospy.Subscriber("/hololens2/image_pv_remap", Image, self.rgb_callback)
    # self.info_sub = rospy.Subscriber("/hololens2/camerainfo_lt", CameraInfo, self.info_callback)


    # def rgb_callback(self, msg):
    #     try:
    #         self.rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")  
    #     except Exception as e:
    #         rospy.logerr(f"Failed to convert pv image: {str(e)}")

    # def depth_callback(self, img_msg):
    #     try:
    #         self.depth_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="32FC1")
    #     except Exception as e:
    #         rospy.logerr(f"Failed to convert depth image: {str(e)}")

    # def info_callback(self, info_msg):
    #     self.camera_info = info_msg