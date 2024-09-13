import argparse

import numpy as np
import cv2
from cv_bridge import CvBridge
import open3d as o3d
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, Vector3Stamped
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
GRASP_MASK_PADDING = 0
NUM_GRASPS = 5

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

        # # Subscribe to gaze and hand points for POTENTIAL FIELD
        # self.gaze_sub = message_filters.Subscriber('/hololens2/goal_gaze_eet', Vector3Stamped)
        # self.index_sub = message_filters.Subscriber('/hololens2/goal_index_tip', Vector3Stamped)
        # self.middle_sub = message_filters.Subscriber('/hololens2/goal_middle_tip', Vector3Stamped)
        # self.thumb_sub = message_filters.Subscriber('/hololens2/goal_thumb_tip', Vector3Stamped)


        # Set up a TimeSynchronizer (for sync sub)
        # self.ts = message_filters.ApproximateTimeSynchronizer([self.rgb_sub, self.depth_sub, self.info_sub, self.gaze_sub, self.index_sub, self.middle_sub, self.thumb_sub], 10, slop=0.1)
        self.ts = message_filters.TimeSynchronizer([self.rgb_sub, self.depth_sub, self.info_sub], 10)
        self.ts.registerCallback(self.callback)

        # rospy.sleep(10) # wait for the first message to arrive

        self.grasp_mask_sub = rospy.Subscriber('/hololens2/grasp_mask', Image, self.mask_callback)

        self.br = TransformBroadcaster()
        self.emg_sub = rospy.Subscriber('/emg', String, self.emg_callback)
        # self.timer = rospy.Timer(rospy.Duration(3), self.timer_callback)

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
   
    # # Sync callback
    # def callback(self, rgb_msg, depth_msg, info_msg, gaze_msg, index_msg, middle_msg, thumb_msg):
    #     try:
    #         print('Received synchronized messages')
    #         self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8") 
    #         self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
    #         self.camera_info_lt = info_msg 
    #         self.gaze = [gaze_msg.vector.x, gaze_msg.vector.y]
    #         self.index = [index_msg.vector.x, index_msg.vector.y]
    #         self.middle = [middle_msg.vector.x, middle_msg.vector.y]
    #         self.thumb = [thumb_msg.vector.x, thumb_msg.vector.y]
            
    #     except Exception as e:
    #         rospy.logerr(f"(Anygrasp) Error processing synchronized messages: {str(e)}")
    
    # Sync callback
    def callback(self, rgb_msg, depth_msg, info_msg):
        try:
            print('Received synchronized messages')
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "rgb8") 
            self.depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            self.camera_info_lt = info_msg 
            
        except Exception as e:
            rospy.logerr(f"(Anygrasp) Error processing synchronized messages: {str(e)}")

    def emg_callback(self, emg_msg):
        if emg_msg.data == 'hand_close':
            rospy.loginfo('(Anygrasp) Emg trigger receieved')
            if self.is_rgbd_info_ready() and not self.is_grasp_predicted():
                rospy.loginfo('(Anygrasp) Starting grasp prediction...')
                self.predict_grasp() # update self.gg (once)
            rospy.sleep(0.2) # buffer time to set flag

            if self.is_grasp_mask_ready() and self.is_grasp_predicted() and not self.is_filtered_grasp_predicted():
                rospy.loginfo('(Anygrasp) Filtering grasps...')
                self.filter_grasp(use_potential_field=False) # update self.filtered_gg (once)
            rospy.sleep(0.2) # buffer time to set flag

            # if self.is_filtered_grasp_predicted():
            #     if cfgs.debug:
            #         self.visualize_grasps()
            #     self.br_grasps()
            #     rospy.loginfo_once('(Anygrasp) Grasp prediction finished, broadcasting grasp tf...') 

    # def timer_callback(self, event):
    #     if self.is_filtered_grasp_predicted:
    #         if cfgs.debug:
    #             self.visualize_grasps()
    #         self.br_grasps()
    #         rospy.loginfo_once('(Anygrasp) Grasp prediction finished, broadcasting grasp tf...')
    #         self.br_grasps()            

    # def emg_callback(self, emg_msg):
    #     if emg_msg.data == 'hand_close':
    #         rospy.loginfo('(Anygrasp) Emg trigger receieved')
    #         if self.is_rgbd_info_ready() and not self.is_grasp_predicted():
    #             rospy.loginfo('(Anygrasp) Starting grasp prediction...')
    #             self.predict_grasp() # update self.gg (once)
    #         else:
    #             rospy.loginfo(f'(Anygrasp) Grasp prediction already done {self.is_grasp_predicted()} or RGBD info not ready {not self.is_rgbd_info_ready()}')
    #         rospy.sleep(0.2) # buffer time to set flag

    #         if self.is_grasp_mask_ready() and self.is_grasp_predicted() and not self.is_filtered_grasp_predicted():
    #             rospy.loginfo('(Anygrasp) Filtering grasps...')
    #             self.filter_grasp(use_potential_field=False) # update self.filtered_gg (once)
    #         else:
    #             rospy.loginfo(f'(Anygrasp) Grasp mask not ready {self.is_grasp_mask_ready()} or Grasp prediction not done {self.is_grasp_predicted()} or Filtered grasp already done {not self.is_filtered_grasp_predicted()}')
    #         rospy.sleep(0.2) # buffer time to set flag

    #         if self.is_filtered_grasp_predicted():
    #             if cfgs.debug:
    #                 self.visualize_grasps()
    #             self.br_grasps()
    #         if self.is_filtered_grasp_predicted():
    #             rospy.loginfo_once('(Anygrasp) Grasp prediction finished, broadcasting grasp tf...')    

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

   
    def filter_grasp(self, num_grasp=NUM_GRASPS, use_potential_field=True, location_weighting = 1.0, grasp_quality_weighting = 0.3, gaze_to_hand_weighting_ratio = 0.2):
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

            if utils.is_grasp_within_mask(grasp_center_2d, self.grasp_mask, padding = GRASP_MASK_PADDING) or utils.is_grasp_within_mask(grasp_center_offset_2d, self.grasp_mask, padding = GRASP_MASK_PADDING):
                filtered_gg_index.append(i)
                grasp_centers_offset_2d_arr.append(grasp_center_offset_2d)
        filtered_gg = self.gg[filtered_gg_index]
        rospy.loginfo(f'(Anygrasp) Number of filtered grasps based on mask: {len(filtered_gg)}')

        # TODO: test potential field
        if use_potential_field and filtered_gg is not None:
            rospy.loginfo(f'(Anygrasp) Using potential field to adjust grasp scores...')
            # filtered_adjusted_gg = filtered_gg
            location_scores = []
            print('location_weighting:', location_weighting)
            print('grasp_quality_weighting:', grasp_quality_weighting)
            print('total weight:', location_weighting + grasp_quality_weighting)
            normalized_weights = [location_weighting/(location_weighting + grasp_quality_weighting), grasp_quality_weighting/(location_weighting + grasp_quality_weighting)]
            rospy.loginfo('(Anygrasp) Normalized weights for location and grasp quality: ')
            for grasp_center_offset_2d in grasp_centers_offset_2d_arr:
                location_scores.append(self.adjust_grasp_scores(grasp_center_offset_2d, gaze_to_hand_weighting_ratio*location_weighting, (1-gaze_to_hand_weighting_ratio)*location_weighting))
            # normalize the location scores and grasp quality scores 
            normalized_location_scores = location_scores / np.sum(location_scores)
            normalized_grasp_quality_scores = filtered_gg.scores / np.sum(filtered_gg.scores)
            normalized_scores = np.dot(normalized_weights, [normalized_location_scores, normalized_grasp_quality_scores])
            rospy.loginfo(f'(Anygrasp) Original grasp scores: {filtered_gg.scores}')
            rospy.loginfo(f'(Anygrasp) Location scores: {location_scores}')
            rospy.loginfo(f'(Anygrasp) Normalized scores: {normalized_scores}')
            rospy.loginfo(f'(Anygrasp) Coordinates: {grasp_centers_offset_2d_arr}')
            # draw the image with the grasp centers and score labels
                        
            for i in range(len(grasp_centers_offset_2d_arr)):
                cv2.circle(self.rgb_image, grasp_centers_offset_2d_arr[i], 5, (0, 255, 0), -1)
                cv2.putText(self.rgb_image, f'{normalized_scores[i]:.2f}', grasp_centers_offset_2d_arr[i], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imwrite('grasps_scores_test.jpg', self.rgb_image)
            cv2.imshow('Grasps with scores', self.rgb_image)
                    
            filtered_adjusted_gg = filtered_gg
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

    def evaluate_potential_field(self, point_x, point_y, node_x, node_y, attract_grasp):
        """
        Evaluate the potential field at a point (x, y) with mode (u, v) acting as an attractor or repulsor
        """
        # Assume inverse square law
        f = 1 / ((float(point_x) - node_x)**2 + (float(point_y) - node_y)**2)
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
        object_size = np.linalg.norm([w, h]) 
        gaze_location_score = self.evaluate_potential_field(grasp_center_2d[0], grasp_center_2d[1], self.gaze[0], self.gaze[1], True) * object_size**2
        index_location_score = self.evaluate_potential_field(grasp_center_2d[0], grasp_center_2d[1], self.index[0], self.index[1], False) * object_size**2
        middle_location_score = self.evaluate_potential_field(grasp_center_2d[0], grasp_center_2d[1], self.middle[0], self.middle[1], False) * object_size**2
        thumb_location_score = self.evaluate_potential_field(grasp_center_2d[0], grasp_center_2d[1], self.thumb[0], self.thumb[1], False) * object_size**2
        # location scores are now dimensionless, normalize the weights
        factor = (gaze_weighting + fingertip_weighting)
        weights = [gaze_weighting/factor, fingertip_weighting/3/factor, fingertip_weighting/3/factor, fingertip_weighting/3/factor]
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

            # Add waypoints between pregrasp and grasp
            tf_grasp_wp1 = prepare_tf_msg(f'grasp_{i}', f'grasp_{i}'+'wp1',
                                    None,
                                    [0,0,-0.5],
                                    [0,0,0,1])
            
            tf_grasp_wp2 = prepare_tf_msg(f'grasp_{i}', f'grasp_{i}'+'wp2',
                                    None,
                                    [0,0,-0.10],
                                    [0,0,0,1])
            
            tf_pregrasp = prepare_tf_msg(f'grasp_{i}', f'pregrasp_{i}',
                                    None,
                                    [0,0,-0.15],
                                    [0,0,0,1])
            
            tf_list.append(tf_grasp)
            tf_list.append(tf_pregrasp)
            tf_list.append(tf_grasp_wp1)
            tf_list.append(tf_grasp_wp2)
            
        rospy.loginfo_once(f'(Anygrasp) Publishing grasps...')
        self.br.sendTransform(tf_list)

if __name__ == '__main__':
    cfgs = parse_arguments()
    cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))
    handler = GraspPredictor(cfgs)

    while not rospy.is_shutdown():
        if handler.is_filtered_grasp_predicted():
            if cfgs.debug:
                handler.visualize_grasps()
            handler.br_grasps()
            rospy.loginfo_once('(Anygrasp) Grasp prediction finished, broadcasting grasp tf...')

    # rospy.spin()