#!/usr/bin/env python3  
import rclpy
from rclpy.node import Node

import math
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
import numpy as np
import yaml
import PIL.Image as Img
import cv2
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion
from cv_bridge import CvBridge
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

class DynamicMapNode(Node):

    def __init__(self, filename='csl_entrance'):
        super().__init__('dynamic_map_node')

        with open(filename + '.yaml', 'r') as f:
            current_map = yaml.safe_load(f)

        img = Img.open(filename + '.pgm')
        self.map_width, self.map_height = img.width, img.height
        self.map_resolution = current_map['resolution']
        self.origin_x = current_map['origin'][0]
        self.origin_y = current_map['origin'][1]

        self.prox_thre = 0.25  # meters
        self.max_range = 25.0  # meters

        static_map = cv2.imread(filename+'.pgm', 0)
        static_pts = np.array(np.where(static_map == 0)).T
        static_pts[:, [1, 0]] = static_pts[:, [0, 1]]
        static_pts[:, 0] = self.origin_x / self.map_resolution
        static_pts[:, 1] = self.map_height + self.origin_y / self.map_resolution - static_pts[:, 1]
        self.tree = KDTree(static_pts * self.map_resolution)

        self.bridge = CvBridge()
        self.static_pts = static_pts * self.map_resolution

        self.rx = self.ry = self.rtheta = 0.0
        self.loc_ready = False

        self.publisher_pose = self.create_publisher(Float32MultiArray, 'robot_pose', 10)
        self.dyn_pub = self.create_publisher(LaserScan, 'filtered_scan', 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.scan_subscriber = self.create_subscription(LaserScan, 'scan', self.update, 10)

        self.timer = self.create_timer(0.033, self.publish_pose)

    def update(self, msg):
        trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
        x, y = trans.transform.translation.x, trans.transform.translation.y
        orientation = [trans.transform.rotation.x, trans.transform.rotation.y,
                        trans.transform.rotation.z, trans.transform.rotation.w]
        _, _, yaw = euler_from_quaternion(orientation)

        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        points_x = np.cos(angles) * np.array(msg.ranges)
        points_y = np.sin(angles) * np.array(msg.ranges)
        data = np.stack((points_x, points_y), axis=-1)

        R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        points_world = np.dot(data, R.T) + [x, y]

        dists, _ = self.tree.query(points_world, k=1)
        msg.ranges = [range if dist > self.prox_thre else float('inf') for range, dist in zip(msg.ranges, dists)]
        self.dyn_pub.publish(msg)

    def publish_pose(self):
        try:
            trans = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            x, y = trans.transform.translation.x, trans.transform.translation.y
            orientation = [trans.transform.rotation.x, trans.transform.rotation.y,
                           trans.transform.rotation.z, trans.transform.rotation.w]
            _, _, yaw = euler_from_quaternion(orientation)

            pose_data = Float32MultiArray(data=[x, y, yaw])
            pose_publisher = self.create_publisher(Float32MultiArray, 'robot_pose', 10)
            pose_data = Float32MultiArray()
            pose_data.data = [x, y, yaw]
            self.publisher.publish(pose_data)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().info('TF2 lookup exception')

def main(args=None):
    rclpy.init(args=args)
    node = DynamicMapNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
