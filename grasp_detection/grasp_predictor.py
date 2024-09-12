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

WORKSPACE_Z_MIN = 0.5
WORKSPACE_Z_MAX = 1.2


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
        # Initialize AnyGrasp model --------------------------------------------------
        self.anygrasp = AnyGrasp(cfgs)
        self.anygrasp.load_net()
        rospy.loginfo_once('(Anygrasp) Anygrasp model loaded')

        # Initialize variables
        self.depth_image = None
        self.rgb_image = None
        self.camera_info_lt = None

        self.grasp_mask = None

        # Flags
        self.is_visualized = False
        self.start_prediction = None
        
        self.tf_lt_tmp = TransformStamped() # the moment grasp is predicted

        self.gg = None # will store all grasps
        self.filtered_gg = None # will store filtered grasps
        self.cloud = None # will store the point cloud
    
        # ROS ----------------------------------------------------------
        rospy.init_node('Anygrasp', anonymous=True)
        rospy.loginfo_once('(Anygrasp) Anygrasp node initialized')
        
        # Initialize the CvBridge and TF2 listener
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Set up subscribers using message_filters 
        self.rgb_sub = message_filters.Subscriber('/hololens2/goal_image_pv_remap', Image)
        self.depth_sub = message_filters.Subscriber('/hololens2/goal_image_lt', Image)
        self.info_sub = message_filters.Subscriber('/hololens2/goal_camerainfo_lt', CameraInfo)
        self.gaze_sub = message_filters.Subscriber('/hololens2/gaze_eet', String)
        
        self.index_sub = message_filters.Subscriber('/hololens2/goal_index_tip', String)
        self.middle_sub = message_filters.Subscriber('/hololens2/goal_middle_tip', String)
        self.thumb_sub = message_filters.Subscriber('/hololens2/goal_thumb_tip', String)


        # Set up a TimeSynchronizer (for sync sub)
        self.ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub, self.info_sub, self.gaze_sub, self.index_sub, self.middle_sub, self.thumb_sub], 10)
        self.ts.registerCallback(self.callback)

        self.grasp_mask_sub = rospy.Subscriber('/hololens2/grasp_mask', Image, self.mask_callback)

        self.br = TransformBroadcaster()
        self.emg_sub = rospy.Subscriber('/emg', String, self.emg_callback)

    # Check functions --------------------------------------------------    
    def is_grasp_mask_ready(self):
        return self.grasp_mask is not None
    
    def is_rgbd_info_ready(self):
        return self.depth_image is not None and self.rgb_image is not None and self.camera_info_lt is not None
    
    def is_grasp_predicted(self):
        return self.gg is not None

    def is_filtered_grasp_predicted(self):
        return self.filtered_gg is not None
    # Reset functions --------------------------------------------------
    # TODO: figure out the reset logic (only needed in sam2_client to stop publihing goals)
    # TODO: and in mask handler to stop publishing mask

    # Callbacks ---------------------------------------------------------                                           
    def mask_callback(self, mask_msg):
        try:
            self.grasp_mask = self.bridge.imgmsg_to_cv2(mask_msg, "mono8") # binary mask
            rospy.loginfo_once('(Anygrasp) Grasp mask received')
        except Exception as e:
            rospy.logerr(f"(Anygrasp) Error processing grasp mask msg: {str(e)}")  
   
    # Sync callback
    def callback(self, rgb_msg, depth_msg, info_msg, gaze_msg, index_msg, middle_msg, thumb_msg):
        try:
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8") 
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            self.camera_info_lt = info_msg 
            self.gaze = gaze_msg
            self.index = index_msg
            self.middle = middle_msg
            self.thumb = thumb_msg

        except Exception as e:
            rospy.logerr(f"(Anygrasp) Error processing synchronized messages: {str(e)}")

    def emg_callback(self, emg_msg):
        if emg_msg.data == 'hand_close':
            rospy.loginfo('(Anygrasp) Emg trigger receieved, starting grasp prediction...')
            if self.is_rgbd_info_ready() and not self.is_grasp_predicted():
                self.predict_grasp() # update self.gg (once)
            rospy.sleep(0.2) # buffer time to set flag

            if self.is_grasp_mask_ready() and self.is_grasp_predicted() and not self.is_filtered_grasp_predicted():
                self.filter_grasp(use_potential_field=True) # update self.filtered_gg (once)
            rospy.sleep(0.2) # buffer time to set flag

            if self.is_filtered_grasp_predicted():
                if cfgs.debug:
                    self.visualize_grasps()
                self.br_grasps()
            rospy.loginfo_once('(Anygrasp) Grasp prediction finished, broadcasting grasp tf...')    

    # Main functions ----------------------------------------------------
    def predict_grasp(self):
        """
        Predict grasps using the AnyGrasp model
        update: self.gg, self.cloud
        """
       
        # Get the point cloud from RGBD + CameraInfo
        fx, fy, cx, cy, scale = self.camera_info_lt.K[0], self.camera_info_lt.K[4], self.camera_info_lt.K[2], self.camera_info_lt.K[5], 1
        xmap, ymap = np.arange(self.depth_image.shape[1]), np.arange(self.depth_image.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = self.depth_image / scale
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z

        # Set workspace to crop point cloud
        mask = (points_z > WORKSPACE_Z_MIN) & (points_z < WORKSPACE_Z_MAX)
        points = np.stack([points_x, points_y, points_z], axis=-1)
        points = points[mask].astype(np.float32)
        colors = self.rgb_image[mask].astype(np.float32) / 255.0 # change to [0,1] from [0,255]

        # Workspace for grasp predictions (gg : GraspGroup)
        xmin = points_x.min()
        xmax = points_x.max()
        ymin = points_y.min()
        ymax = points_y.max()
        zmin = points_z.min()
        zmax = points_z.max()
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]

        gg, cloud = self.anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)
        if len(gg) == 0:
            print('(Anygrasp) No Grasp detected after collision detection!')
        gg = gg.nms().sort_by_score()

        # Update grasp variables
        self.gg = gg 
        self.cloud = cloud 

        rospy.loginfo(f'(Anygrasp) Number of grasps: {len(self.gg)}')

   
    def filter_grasp(self, num_grasp=5, use_potential_field=True, location_weighting = 1, grasp_quality_weighting = 3, gaze_to_hand_weighting_ratio = 0.8):
        """
        Filter grasps based on the grasp mask
        
        update: self.filtered_gg
        flag: self.is_grasp_filtered
        """
        # Filter grasps based on the grasp mask
        filtered_gg_index = []
        grasp_centers_offset_2d_arr = []
        for i in range(len(self.gg)):
            grasp_center_3d = self.gg[i].translation 
            grasp_center_2d = utils.project_3d_to_2d(grasp_center_3d, self.camera_info_lt) 
            # Get grasp center offset onto object for more accurate mask check
            grasp_center_offset_3d = utils.get_offset_grasp_center_3d(self.gg[i])
            grasp_center_offset_2d = utils.project_3d_to_2d(grasp_center_offset_3d, self.camera_info_lt)
            grasp_centers_offset_2d_arr.append(grasp_center_offset_2d)
            if utils.is_grasp_within_mask(grasp_center_2d, self.grasp_mask) or utils.is_grasp_within_mask(grasp_center_offset_2d, self.grasp_mask):
                filtered_gg_index.append(i)
        filtered_gg = self.gg[filtered_gg_index]

        # TODO: test potential field
        if use_potential_field:
            filtered_adjusted_gg = filtered_gg
            location_scores = []
            normalized_weights = [location_weighting, grasp_quality_weighting]/(location_weighting + grasp_quality_weighting)
            
            for grasp_center_offset_2d in grasp_centers_offset_2d_arr:
                location_scores.append(self.adjust_grasp_scores(grasp_center_offset_2d, gaze_to_hand_weighting_ratio*location_weighting, (1-gaze_to_hand_weighting_ratio)*location_weighting))
            # normalize the location scores and grasp quality scores 
            normalized_location_scores = location_scores / np.sum(location_scores)
            normalized_grasp_quality_scores = filtered_gg.scores / np.sum(filtered_gg.scores)
            normalized_scores = np.dot(normalized_weights, [normalized_location_scores, normalized_grasp_quality_scores])
            for i in range(len(filtered_gg)):
                filtered_adjusted_gg[i].score = normalized_scores[i] 
            # sort the grasps by the adjusted scores
            filtered_adjusted_gg = filtered_adjusted_gg.nms().sort_by_score()    
            filtered_gg = filtered_adjusted_gg

        # Only return the top num_grasp grasps      
        if len(filtered_gg) < num_grasp:
            rospy.loginfo(f'(Anygrasp) Number of filtered grasps is {len(filtered_gg)}, returning all grasps')
            self.filtered_gg = filtered_gg
        else:
            rospy.loginfo(f'(Anygrasp) Number of filtered grasps is {len(filtered_gg)}, returning top {num_grasp} grasps')
            self.filtered_gg = filtered_gg[0:num_grasp]    

    def evaluate_potential_field(point, node, attract_grasp = True):
        """
        Evaluate the potential field at a point (x, y) with mode (u, v) acting as an attractor or repulsor
        """
        # Assume inverse square law
        u, v = point[0], point[1]
        x, y = node[0], node[1]
        f = 1 / ((x - u)**2 + (y - v)**2)
        if attract_grasp:
            return f
        else:
            return -f        

    def adjust_grasp_scores(self, grasp_center_2d, gaze_weighting, fingertip_weighting):
        """
        Adjust the grasp scores based on the distances from gaze and hand points, using an artificial potential field
        """          
        # The adjusted grasp score is k_location * location_score + k_grasp_quality * grasp_quality_score
        # location_score is a weighted superposition of a few potential fields.
        # Assume each potential field is inversely proportional to the distance from its source or sink node 
        # The distances used in calculating each potential field are normalized by the characteristic dimension of the object
        x, y, w, h = cv2.boundingRect(self.grasp_mask)
        object_size = np.linalg([w, h]) 
        gaze_location_score = self.evaluate_potential_field(grasp_center_2d, self.gaze, attract_grasp = True) * object_size**2
        index_location_score = self.evaluate_potential_field(grasp_center_2d, self.index, attract_grasp = False) * object_size**2
        middle_location_score = self.evaluate_potential_field(grasp_center_2d, self.middle, attract_grasp = False) * object_size**2
        thumb_location_score = self.evaluate_potential_field(grasp_center_2d, self.thumb, attract_grasp = False) * object_size**2
        # location scores are now dimensionless, normalize the weights
        weights = [gaze_weighting, fingertip_weighting/3, fingertip_weighting/3, fingertip_weighting/3]/(gaze_weighting + fingertip_weighting)
        location_score = np.dot(weights, [gaze_location_score, index_location_score, middle_location_score, thumb_location_score])
        # grasp_quality_score = grasp_quality_weighting * grasp_score
        return location_score #+ grasp_quality_score
        
    def visualize_grasps(self):
        """
        Visualize all grasps and filtered grasps
        
        flag: self.is_visualized
        """
        # Visualize only once
        if not self.is_visualized:
            self.is_visualized = True

            trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
            cloud = self.cloud
            cloud.transform(trans_mat)

            # Visualize all grasps
            grippers = self.gg.to_open3d_geometry_list()
            for gripper in grippers:
                gripper.transform(trans_mat)
            rospy.loginfo('(Anygrasp) Visualizing all grasps...')
            o3d.visualization.draw_geometries([*grippers, cloud]) 
            # o3d.visualization.draw_geometries([grippers[0], cloud]) #visualize the best grasp

            # Visualize filtered grasps
            filtered_grippers = self.filtered_gg.to_open3d_geometry_list()
            for filtered_gripper in filtered_grippers:
                filtered_gripper.transform(trans_mat)  
            rospy.loginfo('(Anygrasp) Visualizing filtered grasps...')
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
            
            tf_list.append(tf_grasp)

        rospy.loginfo_once(f'(Anygrasp) Publishing grasps...')
        self.br.sendTransform(tf_list)

if __name__ == '__main__':
    cfgs = parse_arguments()
    cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))
    handler = GraspPredictor(cfgs)
    rospy.spin()