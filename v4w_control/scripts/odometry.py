#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import rospy
import tf2_ros
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import TransformStamped
from imu_processor import ImuProcessor
from utils import angle_to_quaternion, quaternion_to_angle

class OdometryNode:
    def __init__(self):
        rospy.init_node('odometry_node', anonymous=True)
        
        # Get robot name parameter
        self.robot_name = rospy.get_param('~robot_name', 'v4w1')
        
        # Initialize TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        
        self.witIMU = ImuProcessor(imu_topic="/imu",             # Initialize IMU processor
                              mag_topic="/mag", 
                              sampling_rate=800,
                              method='madgwick', use_mag=False, tait_bryan=False)     
        
        # Initialize subscribers
        self.odom_sub = rospy.Subscriber(
            f'natnet_ros/{self.robot_name}/odom', 
            Odometry, 
            self.odom_callback
        )
        
        self.control_sub = rospy.Subscriber(
            f'{self.robot_name}/control', 
            Float32MultiArray, 
            self.control_callback
        )
        
        rospy.loginfo(f"Odometry node initialized for robot: {self.robot_name}")
        rospy.loginfo(f"Subscribed to:")
        rospy.loginfo(f"  - natnet_ros/{self.robot_name}/odom")
        rospy.loginfo(f"  - /imu")
        rospy.loginfo(f"  - {self.robot_name}/control")
        rospy.loginfo(f"Publishing TF: odom -> {self.robot_name}/base_link")
    
    def odom_callback(self, msg):
        """Handle odometry messages from motion capture system"""
        rospy.logdebug(f"Received odom: pos=({msg.pose.pose.position.x:.3f}, "
                      f"{msg.pose.pose.position.y:.3f}, {msg.pose.pose.position.z:.3f})")
        
        # Create and publish TF transform from odom to base_link
        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = f"{self.robot_name}/base_link"
        
        # Copy position from odometry message
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z
        
        # Copy orientation from odometry message
        transform.transform.rotation.x = msg.pose.pose.orientation.x
        transform.transform.rotation.y = msg.pose.pose.orientation.y
        transform.transform.rotation.z = msg.pose.pose.orientation.z
        transform.transform.rotation.w = msg.pose.pose.orientation.w
        
        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)

        robot_pitch = quaternion_to_angle(transform.transform.rotation)[1]

        # Create and publish TF transform from camera_pivot to camera_base
        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = f"{self.robot_name}/camera_pivot"
        transform.child_frame_id = f"{self.robot_name}/camera_base"
        
        # Copy position from odometry message
        transform.transform.translation.x = 0.013
        transform.transform.translation.y = 0
        transform.transform.translation.z = 0

        pitch_camera = self.witIMU.angles[1]
        transform.transform.rotation = angle_to_quaternion([0.0, -(pitch_camera+robot_pitch), 0.0])
        
        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)

    
    def control_callback(self, msg):
        """Handle control command messages"""
        rospy.logdebug(f"Received control command: {msg.data}")
        # Process control commands here
        pass
    
    def run(self):
        """Main loop"""
        rospy.spin()

if __name__ == '__main__':
    try:
        node = OdometryNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Odometry node shutting down")
    except Exception as e:
        # print all traceback
        import traceback
        traceback.print_exc()