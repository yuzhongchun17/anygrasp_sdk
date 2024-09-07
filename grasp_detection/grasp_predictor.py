import argparse

import numpy as np
import cv2
from cv_bridge import CvBridge
import open3d as o3d

import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool, String
from tf2_ros import TransformBroadcaster, TransformListener
import tf2_ros
from tf.transformations import quaternion_from_euler, quaternion_multiply
import message_filters

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup
# TODO: make this import directly from hololens_ss ros package
from hl2ss_ros_utils import prepare_tf_msg

import grasp_predictor_utils as utils

# Parse terminal input
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
    parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
    parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
    parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()


class GraspPredictor:
    def __init__(self, cfgs):
        self.cfgs = cfgs

        # Initialize variables
        self.depth_image = None
        self.rgb_image = None
        self.camera_info_lt = None

        self.grasp_mask = None

        # Flags
        self.is_grasp_filtered = False
        self.is_visualized = False
        self.start_prediction = None
        
        self.tf_lt_tmp = TransformStamped() # the moment grasp is predicted

        self.gg = None # will store all grasps
        self.filtered_gg = None # will store filtered grasps
        self.cloud = None # will store the point cloud
    
        rospy.init_node('Anygrasp', anonymous=True)
        rospy.loginfo_once('Anygrasp node initialized')
        # ROS ----------------------------------------------------------
        # Initialize the CvBridge and TF2 listener
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Set up subscribers using message_filters 
        self.rgb_sub = message_filters.Subscriber('/hololens2/goal_image_pv_remap', Image)
        self.depth_sub = message_filters.Subscriber('/hololens2/goal_image_lt', Image)
        self.info_sub = message_filters.Subscriber('/hololens2/goal_camerainfo_lt', CameraInfo)

        # Set up a TimeSynchronizer (for sync sub)
        self.ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub, self.info_sub], 10)
        self.ts.registerCallback(self.callback)

        self.br = TransformBroadcaster()
        self.timer = rospy.Timer(rospy.Duration(0.5), self.timer_callback)  # Adjust duration as needed

        self.grasp_mask_sub = rospy.Subscriber('/hololens2/grasp_mask', Image, self.mask_callback)

    # Check functions --------------------------------------------------    
    def is_grasp_mask_ready(self):
        return self.grasp_mask is not None
    
    def is_rgbd_info_ready(self):
        return self.depth_image is not None and self.rgb_image is not None and self.camera_info_lt is not None
    
    def is_grasp_predicted(self):
        return self.gg is not None

    def is_filtered_grasp_predicted(self):
        return self.filtered_gg is not None
    
    # Callbacks ---------------------------------------------------------                                           
    def mask_callback(self, mask_msg):
        try:
            self.grasp_mask = self.bridge.imgmsg_to_cv2(mask_msg, "mono8") # binary mask
            rospy.loginfo_once('Grasp mask received')
        except Exception as e:
            rospy.logerr(f"Error processing grasp mask msg: {str(e)}")
   
    # Sync callback
    def callback(self, rgb_msg, depth_msg, info_msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8") 
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            self.camera_info_lt = info_msg 

        except Exception as e:
            rospy.logerr(f"Error processing synchronized messages: {str(e)}")

    def timer_callback(self, event):
        rospy.loginfo_once('Timer callback triggered')
        if self.is_rgbd_info_ready() and not self.is_grasp_predicted():
            # NOTE: RGBD ready, but grasp not predicted yet
            # utils.visualize_pcd(self.rgb_image, self.depth_image, self.camera_info)
            self.predict_grasp() # update self.gg (once)

        if self.is_grasp_mask_ready() and self.is_grasp_predicted() and not self.is_filtered_grasp_predicted():
            # NOTE: Grasp mask ready, grasps predicted, but not filtered yet
            self.filter_grasp() # update self.filtered_gg (once)

        if self.is_filtered_grasp_predicted():
            if cfgs.debug:
                self.visualize_grasps()
            self.br_grasps()

    # Main functions ----------------------------------------------------
    def predict_grasp(self):
        """
        Predict grasps using the AnyGrasp model
        update: self.gg, self.cloud
        """
        anygrasp = AnyGrasp(cfgs)
        anygrasp.load_net()
       
        # get camera parameters
        fx, fy, cx, cy, scale = self.camera_info_lt.K[0], self.camera_info_lt.K[4], self.camera_info_lt.K[2], self.camera_info_lt.K[5], 1
        # get point cloud
        xmap, ymap = np.arange(self.depth_image.shape[1]), np.arange(self.depth_image.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = self.depth_image / scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        # set your workspace to crop point cloud
        mask = (points_z > 0.3) & (points_z < 1)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = self.rgb_image[mask].astype(np.float32) / 255.0 # change to [0,1] from [0,255]
        # print(points.min(axis=0), points.max(axis=0))

        # Workspace for grasp predictions (gg : GraspGroup)
        xmin = points_x.min()
        xmax = points_x.max()
        ymin = points_y.min()
        ymax = points_y.max()
        zmin = points_z.min()
        zmax = points_z.max()
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)
        if len(gg) == 0:
            print('No Grasp detected after collision detection!')
        gg = gg.nms().sort_by_score()

        self.gg = gg # update all predicted grasps
        self.cloud = cloud # update the point cloud

        rospy.loginfo(f'Number of grasps: {len(self.gg)}')

    
    def filter_grasp(self, num_grasp=5):
        """
        Filter grasps based on the grasp mask
        
        update: self.filtered_gg
        flag: self.is_grasp_filtered
        """
        if not self.is_grasp_filtered:
            # Filter grasps based on the grasp mask
            filtered_gg_index = []
            for i in range(len(self.gg)):
                grasp_center_3d = self.gg[i].translation 
                grasp_center_2d = utils.project_3d_to_2d(grasp_center_3d, self.camera_info_lt) 
                if utils.is_grasp_within_mask(grasp_center_2d, self.grasp_mask):
                    filtered_gg_index.append(i)
            filtered_gg = self.gg[filtered_gg_index]

            # Only return the top num_grasp grasps      
            if len(filtered_gg) < num_grasp:
                rospy.loginfo(f'Number of filtered grasps is {len(filtered_gg)}, returning all grasps')
                self.filtered_gg = filtered_gg
            else:
                rospy.loginfo(f'Number of filtered grasps is {len(filtered_gg)}, returning top {num_grasp} grasps')
                self.filtered_gg = filtered_gg[0:num_grasp]          

            # Set the flag
            self.is_grasp_filtered = True

    def visualize_grasps(self):
        """
        Visualize all grasps and filtered grasps
        
        flag: self.is_visualized
        """
        # run visualization only once
        if not self.is_visualized:
            self.is_visualized = True

            trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            cloud = self.cloud
            cloud.transform(trans_mat)

            # visualize all grasps
            grippers = self.gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
            rospy.loginfo('Visualizing all grasps...')
            o3d.visualization.draw_geometries([*grippers, cloud]) 
            # o3d.visualization.draw_geometries([grippers[0], cloud]) #visualize the best grasp

            # visualize filtered grasps
            filtered_grippers = self.filtered_gg.to_open3d_geometry_list()
            for filtered_gripper in filtered_grippers:
                filtered_gripper.transform(trans_mat)  
            rospy.loginfo('Visualizing filtered grasps...')
            o3d.visualization.draw_geometries([*filtered_grippers, cloud]) 

    def br_grasps(self):
        """
        Publish all the grasps and pregrasps as tf relative to lt_tmp
        """
        # Prepare pregrasp and grasp tf messages
        R_cam_ee = utils.get_cam_ee_rotation()
        tf_list = []
        for i, grasp in enumerate(self.filtered_gg):
            rot_in_ee_frame = grasp.rotation_matrix @ R_cam_ee
            tf_grasp = prepare_tf_msg('lt_tmp', f'grasp_{i}', 
                                      None, 
                                      grasp.translation, 
                                      rot_in_ee_frame)
            
            tf_pregrasp = prepare_tf_msg(f'grasp_{i}', f'pregrasp_{i}',
                                        None,
                                        [0,0,-0.10],
                                        [0,0,0,1])
            tf_list.append(tf_grasp)
            tf_list.append(tf_pregrasp)

        rospy.loginfo_once(f'Publishing grasps...')
        self.br.sendTransform(tf_list)

if __name__ == '__main__':
    cfgs = parse_arguments()
    cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))
    handler = GraspPredictor(cfgs)
    rospy.spin()