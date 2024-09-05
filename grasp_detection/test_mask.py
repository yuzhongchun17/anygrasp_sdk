#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import numpy as np
import cv2  # Import OpenCV for drawing the ball

def publish_dummy_mask():
    rospy.init_node('dummy_mask_publisher', anonymous=True)
    pub = rospy.Publisher('/hololens2/grasp_mask', Image, queue_size=10)
    rate = rospy.Rate(10)  # 10hz

    # Image dimensions and properties
    dummy_height = 288
    dummy_width = 320
    dummy_channels = 1  # Grayscale image

    # Define a medium-sized white ball
    ball_radius = dummy_height // 3
    center_x, center_y = dummy_width // 2, dummy_height // 2  # Center of the image

    while not rospy.is_shutdown():
        # Create a blank black image
        dummy_mask = np.zeros((dummy_height, dummy_width, dummy_channels), dtype=np.uint8)
        
        # Draw a white ball in the center
        cv2.circle(dummy_mask, (center_x, center_y), ball_radius, (255,), -1)  # 255 for white color, -1 for filled circle

        # Prepare the ROS message
        msg = Image()
        msg.height = dummy_height
        msg.width = dummy_width
        msg.encoding = 'mono8'
        msg.is_bigendian = 0
        msg.step = dummy_width * dummy_channels
        msg.data = dummy_mask.tobytes()

        pub.publish(msg)
        rospy.loginfo_once("Start publishing the dummy mask ...")
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_dummy_mask()
    except rospy.ROSInterruptException:
        pass
