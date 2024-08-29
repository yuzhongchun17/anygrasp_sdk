#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool

def publish_bool():
    # Initialize the ROS node
    rospy.init_node('bool_publisher', anonymous=True)

    # Create a publisher object that will publish Bool messages on the 'bool_topic' topic
    pub = rospy.Publisher('/grasp_trigger', Bool, queue_size=10)

    # Set the rate of publishing to 1 Hz
    rate = rospy.Rate(1)  # 1 Hz

    # Log information once
    rospy.loginfo_once("Starting to publish boolean values")

    while not rospy.is_shutdown():
        # Create a Bool message instance
        bool_msg = Bool()
        bool_msg.data = True  # Set the data field to True

        # Publish the message
        pub.publish(bool_msg)

        # Log the published message once
        rospy.loginfo_once("Published Bool: True")

        # Sleep to maintain the publishing rate
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_bool()
    except rospy.ROSInterruptException:
        pass
