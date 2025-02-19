#!/usr/bin/env python3
import argparse
from math import sin, cos
import numpy as np

import rclpy
import rclpy.serialization
from rclpy.serialization import serialize_message
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from builtin_interfaces.msg import Time as BuiltinTime

def float_to_time_msg(t_float):
    sec = int(t_float)
    nanosec = int((t_float - sec) * 1e9)
    t_msg = BuiltinTime()
    t_msg.sec = sec
    t_msg.nanosec = nanosec
    return t_msg

def load_scans(fname):
    data = np.genfromtxt(fname, delimiter=",")
    seqs = data[:, 0].astype(np.uint32)
    times = data[:, 1].astype(np.float32)
    scans = data[:, 2:].astype(np.float32)
    return seqs, times, scans

def load_odoms(fname):
    data = np.genfromtxt(fname, delimiter=",")
    seqs = data[:, 0].astype(np.uint32)
    times = data[:, 1].astype(np.float32)
    odos = data[:, 2:].astype(np.float32)   # x, y, phi
    return seqs, times, odos

def sequence_to_bag(seq_fname, bag_dir):
    # Create a template LaserScan message.
    scan_msg = LaserScan()
    scan_msg.header.frame_id = 'sick_laser_front'
    scan_msg.angle_min = np.radians(-225.0 / 2)
    scan_msg.angle_max = np.radians(225.0 / 2)
    scan_msg.range_min = 0.005
    scan_msg.range_max = 100.0
    scan_msg.scan_time = 0.066667
    scan_msg.time_increment = 0.000062
    scan_msg.angle_increment = (scan_msg.angle_max - scan_msg.angle_min) / 450

    # Create a template TransformStamped (used in a TFMessage).
    tran = TransformStamped()
    tran.header.frame_id = 'base_footprint'
    tran.child_frame_id = 'sick_laser_front'

    # Set up rosbag2_py writer.
    storage_options = StorageOptions(uri=bag_dir, storage_id='sqlite3')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    writer = SequentialWriter()
    writer.open(storage_options, converter_options)

    # Write LaserScan messages.
    seqs, times, scans = load_scans(seq_fname)
    for seq, t_val, scan in zip(seqs, times, scans):
        t_msg = float_to_time_msg(t_val)
        scan_msg.header.seq = int(seq)
        scan_msg.header.stamp = t_msg
        # Convert the numpy array of ranges to a Python list.
        scan_msg.ranges = scan.tolist()
        serialized = serialize_message(scan_msg)
        # rosbag2_py expects the timestamp in nanoseconds.
        timestamp_ns = int(t_val * 1e9)
        writer.write('/sick_laser_front/scan', serialized, timestamp_ns)

    # Write odometry data as TF messages.
    # Assumes the odometry file is named similarly to the scan file,
    # with the last three characters replaced by 'odom2'.
    odom_fname = seq_fname[:-3] + 'odom2'
    seqs_odom, times_odom, odoms = load_odoms(odom_fname)
    for seq, t_val, odom in zip(seqs_odom, times_odom, odoms):
        t_msg = float_to_time_msg(t_val)
        tran.header.seq = int(seq)
        tran.header.stamp = t_msg
        tran.transform.translation.x = float(odom[0])
        tran.transform.translation.y = float(odom[1])
        tran.transform.translation.z = 0.0
        tran.transform.rotation.x = 0.0
        tran.transform.rotation.y = 0.0
        tran.transform.rotation.z = sin(odom[2] * 0.5)
        tran.transform.rotation.w = cos(odom[2] * 0.5)
        tf_msg = TFMessage()
        tf_msg.transforms = [tran]
        serialized = serialize_message(tf_msg)
        timestamp_ns = int(t_val * 1e9)
        writer.write('/tf', serialized, timestamp_ns)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sequence to ROS2 bag converter")
    parser.add_argument("--seq", type=str, required=True, help="Path to sequence file (CSV)")
    parser.add_argument("--output", type=str, required=False, default="./out_bag", help="Output bag directory")
    args = parser.parse_args()

    rclpy.init(args=args)
    sequence_to_bag(args.seq, args.output)
    rclpy.shutdown()
