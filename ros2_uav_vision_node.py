#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped, PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

# Import de la classe principale de vision
from uav_vision_suite.drone_vision_tracker import DroneVisionTracker


class UAVVisionNode(Node):
    def __init__(self):
        super().__init__('uav_vision_node')

        # 1. Initialisation du moteur de vision
        self.tracker = DroneVisionTracker()
        self.bridge = CvBridge()

        # 2. Souscription au flux caméra Gazebo / PX4 SITL
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # 3. Publication des commandes de vitesse vers MAVROS / PX4
        self.vel_pub = self.create_publisher(
            TwistStamped,
            '/mavros/setpoint_velocity/cmd_vel_unstamped',
            10
        )

        # 4. Publication de l'odométrie visuelle pour l'EKF2 de PX4
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/mavros/vision_pose/pose',
            10
        )

        # 5. Publication du flux vidéo HUD traité
        self.hud_pub = self.create_publisher(
            Image,
            '/uav/hud_output',
            10
        )

        self.get_logger().info("ROS 2 UAV Vision & Guidance Node initialized.")

    def euler_to_quaternion(self, roll_deg, pitch_deg, yaw_deg):
        """Convertit les angles d'Euler (degrés) en Quaternion pour ROS 2."""
        r = math.radians(roll_deg)
        p = math.radians(pitch_deg)
        y = math.radians(yaw_deg)

        cy = math.cos(y * 0.5)
        sy = math.sin(y * 0.5)
        cp = math.cos(p * 0.5)
        sp = math.sin(p * 0.5)
        cr = math.cos(r * 0.5)
        sr = math.sin(r * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return qx, qy, qz, qw

    def image_callback(self, msg):
        try:
            # Conversion ROS Image -> OpenCV BGR
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion error: {e}")
            return

        # Traitement de l'image par le moteur de vision
        processed_frame, status = self.tracker.process_frame(frame)

        # A. Publication de la consigne de vitesse (Visual Servoing)
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base_link'
        
        twist_msg.twist.linear.x = float(self.tracker.guidance_cmd['vx'])
        twist_msg.twist.linear.y = float(self.tracker.guidance_cmd['vy'])
        twist_msg.twist.linear.z = float(self.tracker.guidance_cmd['vz'])
        twist_msg.twist.angular.z = float(self.tracker.guidance_cmd['yaw_rate'])
        
        self.vel_pub.publish(twist_msg)

        # B. Publication de la pose visuelle 6-DoF pour PX4 EKF2
        pose_msg = PoseStamped()
        pose_msg.header.stamp = twist_msg.header.stamp
        pose_msg.header.frame_id = 'map'
        
        pose_msg.pose.position.x = float(self.tracker.cur_t[0][0])
        pose_msg.pose.position.y = float(self.tracker.cur_t[1][0])
        pose_msg.pose.position.z = float(self.tracker.cur_t[2][0])

        qx, qy, qz, qw = self.euler_to_quaternion(
            self.tracker.euler_angles[0],
            self.tracker.euler_angles[1],
            self.tracker.euler_angles[2]
        )
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz
        pose_msg.pose.orientation.w = qw

        self.pose_pub.publish(pose_msg)

        # C. Publication du flux d'affichage HUD
        hud_msg = self.bridge.cv2_to_imgmsg(processed_frame, encoding='bgr8')
        hud_msg.header.stamp = twist_msg.header.stamp
        self.hud_pub.publish(hud_msg)


def main(args=None):
    rclpy.init(args=args)
    node = UAVVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.tracker.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()