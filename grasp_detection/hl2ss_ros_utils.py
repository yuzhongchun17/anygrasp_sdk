import rospy
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from tf_conversions import transformations 
from tf.transformations import quaternion_from_matrix

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np


# ------------------------- Prepare msg data-----------------------------
def prepare_Image(image, frame_id, desired_encoding='passthrough', stamp=None):
    """
    Prepares an Image message from an image array.
    
    Parameters:
        image (np.array): The image data array.
        frame_id (str): The frame ID to set in the message header.
        desired_encoding (str): The encoding type for the image (e.g., 'bgr8', 'mono8').
        stamp (rospy.Time, optional): The timestamp for the header. Defaults to current time if None.

    Returns:
        sensor_msgs/Image: A ROS Image message.
    """
    # Define data
    bridge = CvBridge()
    image_msg = bridge.cv2_to_imgmsg(image, encoding=desired_encoding)

    # Define header
    if stamp is None:
        stamp = rospy.Time.now()  # Set default value to current time if not provided
    image_msg.header.stamp = stamp
    image_msg.header.frame_id = frame_id
   
    return image_msg

def prepare_CameraInfo(cam_K, frame_id, cam_height, cam_width, stamp=None):
    """
    Prepares a CameraInfo message.

    Parameters:
        cam_K (list): The camera matrix [K] as a list of 9 elements.
        frame_id (str): The frame ID to set in the message header.
        image_height (int): The height of the image.
        image_width (int): The width of the image.
        stamp (rospy.Time, optional): The timestamp for the header. Defaults to current time if None.

    Returns:
        CameraInfo: A ROS CameraInfo message.
    """
    camera_info = CameraInfo()
    # Define header
    if stamp is None:
        stamp = rospy.Time.now()  # Set default value to current time if not provided
    camera_info.header.stamp = stamp
    camera_info.header.frame_id = frame_id

    # Define other info
    camera_info.height = cam_height
    camera_info.width = cam_width
    camera_info.K = cam_K
    return camera_info

# ------------------------- Prepare tf msg -----------------------------
def prepare_tf_msg(frame_id: str, child_frame_id: str, stamp=None, *args):
    """
    Prepares a TransformStamped message from various input types.
    
    :param frame_id: The ID of the reference frame.
    :param child_frame_id: The ID of the child frame.
    :param args: Variable arguments which can be:
        - (translation, rotation) where translation is [x, y, z] and rotation is 3x3 matrix or quaternion [x, y, z, w].
        - (transformation_matrix) where transformation_matrix is a 4x4 matrix.
    :return: TransformStamped message.
    """
    tf_msg = TransformStamped()
    # Define header
    if stamp is None:
        stamp = rospy.Time.now()
    tf_msg.header.stamp = stamp
    tf_msg.header.frame_id = frame_id
    tf_msg.child_frame_id = child_frame_id
    
    if len(args) == 1 and isinstance(args[0], np.ndarray) and args[0].shape == (4, 4):
        # Case 1: 4x4 transformation matrix
        matrix = args[0]
        translation = matrix[:3, 3]
        rotation_matrix = matrix[:3, :3]
        quaternion = quaternion_from_matrix(matrix)  # Directly use 4x4 matrix
    elif len(args) == 2:
        translation = args[0]
        if isinstance(args[1], np.ndarray) and args[1].shape == (3, 3):
            # Case 2: translation + rotation matrix
            rotation_matrix = np.eye(4)
            rotation_matrix[:3, :3] = args[1]
            quaternion = quaternion_from_matrix(rotation_matrix)  # Use 4x4 matrix
        elif isinstance(args[1], list) or isinstance(args[1], np.ndarray) and len(args[1]) == 4:
            # Case 3: translation + quaternion
            quaternion = args[1]
        else:
            raise ValueError("Invalid rotation data type or size.")
    else:
        raise ValueError("Invalid arguments for TF preparation.")
    
    # Set translation
    tf_msg.transform.translation.x = translation[0]
    tf_msg.transform.translation.y = translation[1]
    tf_msg.transform.translation.z = translation[2]
    
    # Set rotation
    tf_msg.transform.rotation.x = quaternion[0]
    tf_msg.transform.rotation.y = quaternion[1]
    tf_msg.transform.rotation.z = quaternion[2]
    tf_msg.transform.rotation.w = quaternion[3]
    
    return tf_msg

# ------------------------ change camera frame to OpenCV -----------------------------
# lt frame to OpenCV frame
def camera_pose_lt_to_cv(camera_pose):
    """
    Converts a camera pose from lt coordinates to OpenCV coordinates (for AnyGrasp)

    Parameters:
        camera_pose (np.array): The camera pose in the world frame in OpenGL coordinates.

    Returns:
        np.array: The camera pose in the world frame in OpenCV coordinates.
    """
    T_w_lt = camera_pose
    T_lt_cv = np.array([[0, 1, 0, 0],
                        [-1, 0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])  # Transformation matrix from lt to OpenCV coordinates
    T_w_cv = T_w_lt @ T_lt_cv  # Convert to OpenCV coordinates
    return T_w_cv

# pv (OpenGl) to OpenCV frame
def camera_pose_gl_to_cv(camera_pose):
    """
    Converts a camera pose from OpenGL coordinates to OpenCV coordinates.

    Parameters:
        camera_pose (np.array): The camera pose in the world frame in OpenGL coordinates.

    Returns:
        np.array: The camera pose in the world frame in OpenCV coordinates.
    """
    T_w_gl = camera_pose  # Camera pose in OpenGL coordinates to world frame (OpenGL)
    T_gl_cv = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])  # Transformation matrix from OpenGL to OpenCV coordinates
    T_w_cv = T_w_gl @ T_gl_cv  # Convert to OpenCV coordinates
    return T_w_cv
