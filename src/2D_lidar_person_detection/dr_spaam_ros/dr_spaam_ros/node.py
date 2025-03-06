#!/usr/bin/env python3
import numpy as np
from math import sin, cos
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point, Pose, PoseArray
from visualization_msgs.msg import Marker

from dr_spaam.detector import Detector  # Assuming this module is ROS2–compatible

class DrSpaamROS(Node):
    """ROS2 node to detect pedestrians using DROW3 or DR-SPAAM."""

    def __init__(self):
        super().__init__('dr_spaam_ros')
        self._declare_parameters()
        self._read_params()

        self._detector = Detector(
            self.weight_file,
            model=self.detector_model,
            gpu=True,
            stride=self.stride,
            panoramic_scan=self.panoramic_scan,
        )
        self._init_publish_subscribe()

    def _declare_parameters(self):
        # Declare your parameters (set default values as needed)
        self.declare_parameter('weight_file', '/home/sora/colcon_ws/src/2D_lidar_person_detection/trained_models/ckpt_jrdb_ann_dr_spaam_e20.pth')
        self.declare_parameter('conf_thresh', 0.5)
        self.declare_parameter('stride', 1)
        self.declare_parameter('detector_model', 'DR-SPAAM')
        self.declare_parameter('panoramic_scan', False)
        # Parameters for subscribers
        self.declare_parameter('subscriber.scan.topic', '/scan')
        self.declare_parameter('subscriber.scan.queue_size', 10)
        # Parameters for publishers
        self.declare_parameter('publisher.detections.topic', '/detections')
        self.declare_parameter('publisher.detections.queue_size', 10)
        self.declare_parameter('publisher.detections.latch', True)
        self.declare_parameter('publisher.rviz.topic', '/rviz')
        self.declare_parameter('publisher.rviz.queue_size', 10)
        self.declare_parameter('publisher.rviz.latch', True)

    def _read_params(self):
        self.weight_file = self.get_parameter('weight_file').value
        self.conf_thresh = self.get_parameter('conf_thresh').value
        self.stride = self.get_parameter('stride').value
        self.detector_model = self.get_parameter('detector_model').value
        self.panoramic_scan = self.get_parameter('panoramic_scan').value

    def _init_publish_subscribe(self):
        from rclpy.qos import QoSProfile, DurabilityPolicy

        # Publishers with transient_local QoS to mimic latch behavior
        dets_topic = self.get_parameter('publisher.detections.topic').value
        dets_qos = QoSProfile(
            depth=self.get_parameter('publisher.detections.queue_size').value,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self._dets_pub = self.create_publisher(PoseArray, dets_topic, dets_qos)

        rviz_topic = self.get_parameter('publisher.rviz.topic').value
        rviz_qos = QoSProfile(
            depth=self.get_parameter('publisher.rviz.queue_size').value,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        self._rviz_pub = self.create_publisher(Marker, rviz_topic, rviz_qos)

        # Subscriber for LaserScan
        scan_topic = self.get_parameter('subscriber.scan.topic').value
        scan_queue_size = self.get_parameter('subscriber.scan.queue_size').value
        self._scan_sub = self.create_subscription(
            LaserScan, scan_topic, self._scan_callback, scan_queue_size
        )

    def _scan_callback(self, msg: LaserScan):
        # If no one is listening, skip processing
        if (self._dets_pub.get_subscription_count() == 0 and
            self._rviz_pub.get_subscription_count() == 0):
            return

        if not self._detector.is_ready():
            fov_deg = np.rad2deg(msg.angle_increment * len(msg.ranges))
            self._detector.set_laser_fov(fov_deg)

        scan = np.array(msg.ranges)
        scan[scan == 0.0] = 29.99
        scan[np.isinf(scan)] = 29.99
        scan[np.isnan(scan)] = 29.99

        dets_xy, dets_cls, _ = self._detector(scan)
        conf_mask = (dets_cls >= self.conf_thresh).reshape(-1)
        dets_xy = dets_xy[conf_mask]
        dets_cls = dets_cls[conf_mask]

        dets_msg = detections_to_pose_array(dets_xy, dets_cls)
        # Copy header from incoming message
        dets_msg.header = msg.header
        self._dets_pub.publish(dets_msg)

        rviz_msg = detections_to_rviz_marker(dets_xy, dets_cls)
        rviz_msg.header = msg.header
        self._rviz_pub.publish(rviz_msg)


def detections_to_rviz_marker(dets_xy, dets_cls):
    """
    Convert detections to an RViz Marker message. Each detection is marked as a circle
    approximated by a series of line segments.
    """
    msg = Marker()
    msg.action = Marker.ADD
    msg.ns = "dr_spaam_ros"
    msg.id = 0
    msg.type = Marker.LINE_LIST

    # Set identity orientation
    msg.pose.orientation.x = 0.0
    msg.pose.orientation.y = 0.0
    msg.pose.orientation.z = 0.0
    msg.pose.orientation.w = 1.0

    msg.scale.x = 0.03  # line width
    msg.color.r = 1.0
    msg.color.a = 1.0

    r = 0.4
    ang = np.linspace(0, 2 * np.pi, 20)
    xy_offsets = r * np.stack((np.cos(ang), np.sin(ang)), axis=1)

    for d_xy, _ in zip(dets_xy, dets_cls):
        for i in range(len(xy_offsets) - 1):
            p0 = Point()
            p0.x = d_xy[0] + xy_offsets[i, 0]
            p0.y = d_xy[1] + xy_offsets[i, 1]
            p0.z = 0.0
            msg.points.append(p0)

            p1 = Point()
            p1.x = d_xy[0] + xy_offsets[i + 1, 0]
            p1.y = d_xy[1] + xy_offsets[i + 1, 1]
            p1.z = 0.0
            msg.points.append(p1)
    return msg


def detections_to_pose_array(dets_xy, dets_cls):
    pose_array = PoseArray()
    for d_xy, _ in zip(dets_xy, dets_cls):
        p = Pose()
        p.position.x = d_xy[0]
        p.position.y = d_xy[1]
        p.position.z = 0.0
        pose_array.poses.append(p)
    return pose_array


def main(args=None):
    rclpy.init(args=args)
    node = DrSpaamROS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
