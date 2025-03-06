import json
import numpy as np
import os
import shutil
import rosbag2_py
import rclpy.serialization
from sensor_msgs.msg import LaserScan

# Set root dir to JRDB
_jrdb_dir = "./data/JRDB"

# Variables defining output location.
# Do not change unless for you know what you are doing.
_output_laser_dir_name = "lasers"
_output_frames_laser_im_fname = "frames_img_laser.json"
_output_frames_laser_pc_fname = "frames_pc_laser.json"
_output_laser_timestamp_fname = "timestamps.txt"


def _laser_idx_to_fname(idx):
    return str(idx).zfill(6) + ".txt"


def extract_laser_from_rosbag(split):
    """Extract and save combined laser scan from a ROS2 bag.
       Existing files will be overwritten.
    """
    data_dir = os.path.join(_jrdb_dir, split + "_dataset")
    timestamp_dir = os.path.join(data_dir, "timestamps")
    bag_dir = os.path.join(data_dir, "rosbags")
    sequence_names = os.listdir(timestamp_dir)

    laser_dir = os.path.join(data_dir, _output_laser_dir_name)
    if os.path.exists(laser_dir):
        shutil.rmtree(laser_dir)
    os.mkdir(laser_dir)

    for idx, seq_name in enumerate(sequence_names):
        seq_laser_dir = os.path.join(laser_dir, seq_name)
        os.mkdir(seq_laser_dir)

        # In ROS2 the bag is a directory. Here we assume the bag directory
        # is named with a ".bag" extension as before.
        bag_path = os.path.join(bag_dir, seq_name + ".bag")
        print("({}/{}) Extract laser from {} to {}".format(
            idx + 1, len(sequence_names), bag_path, seq_laser_dir)
        )

        # Set up rosbag2_py reader for ROS2
        storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr', output_serialization_format='cdr'
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        # extract all laser messages on the topic "segway/scan_multi"
        timestamp_list = []
        count = 0
        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic == "segway/scan_multi":
                # Deserialize the message using the LaserScan type
                msg = rclpy.serialization.deserialize_message(data, LaserScan)
                scan = np.array(msg.ranges)
                fname = _laser_idx_to_fname(count)
                np.savetxt(os.path.join(seq_laser_dir, fname), scan, newline=" ")
                # Convert ROS2 timestamp (in nanoseconds) to seconds
                timestamp_list.append(t / 1e9)
                count += 1

        np.savetxt(os.path.join(seq_laser_dir, _output_laser_timestamp_fname),
                   np.array(timestamp_list), newline=" ")

        # No explicit close is needed; reader resources are released automatically


def _match_pc_im_laser_one_sequence(split, sequence_name):
    """Write a JSON file that contains URLs for matching pointcloud, laser, and image data.
       Pointclouds are the reference sensor to which others are synchronized.
    """
    data_dir = os.path.join(_jrdb_dir, split + "_dataset")
    timestamp_dir = os.path.join(data_dir, "timestamps", sequence_name)
    laser_dir = os.path.join(data_dir, "lasers", sequence_name)

    # pc frames
    pc_frames_file = os.path.join(timestamp_dir, "frames_pc.json")
    with open(pc_frames_file, "r") as f:
        pc_frames = json.load(f)["data"]

    # im frames
    im_frames_file = os.path.join(timestamp_dir, "frames_img.json")
    with open(im_frames_file, "r") as f:
        im_frames = json.load(f)["data"]

    # synchronize pc and image frames
    pc_timestamp = np.array([float(f["timestamp"]) for f in pc_frames])
    im_timestamp = np.array([float(f["timestamp"]) for f in im_frames])
    pc_im_ft_diff = np.abs(pc_timestamp.reshape(-1, 1) - im_timestamp.reshape(1, -1))
    pc_im_matching_inds = pc_im_ft_diff.argmin(axis=1)

    # synchronize pc and laser frames
    laser_timestamp = np.loadtxt(os.path.join(laser_dir, _output_laser_timestamp_fname), dtype=np.float64)
    pc_laser_ft_diff = np.abs(pc_timestamp.reshape(-1, 1) - laser_timestamp.reshape(1, -1))
    pc_laser_matching_inds = pc_laser_ft_diff.argmin(axis=1)

    # create a merged frame dictionary
    output_frames = []
    for i in range(len(pc_frames)):
        frame = {
            "pc_frame": pc_frames[i],
            "im_frame": im_frames[pc_im_matching_inds[i]],
            "laser_frame": {
                "url": os.path.join(_output_laser_dir_name, sequence_name,
                                    _laser_idx_to_fname(pc_laser_matching_inds[i])),
                "name": "laser_combined",
                "timestamp": laser_timestamp[pc_laser_matching_inds[i]],
            },
            "timestamp": pc_frames[i]["timestamp"],
            "frame_id": pc_frames[i]["frame_id"],
        }

        # adjust file URLs for pointclouds and images
        for pc_dict in frame["pc_frame"]["pointclouds"]:
            f_name = os.path.basename(pc_dict["url"])
            pc_dict["url"] = os.path.join("pointclouds", pc_dict["name"], sequence_name, f_name)

        for im_dict in frame["im_frame"]["cameras"]:
            f_name = os.path.basename(im_dict["url"])
            cam_name = ("image_stitched" if im_dict["name"] == "stitched_image0"
                        else im_dict["name"][:-1] + "_" + im_dict["name"][-1])
            im_dict["url"] = os.path.join("images", cam_name, sequence_name, f_name)

        output_frames.append(frame)

    output_dict = {"data": output_frames}
    f_name = os.path.join(timestamp_dir, "frames_pc_im_laser.json")
    with open(f_name, "w") as fp:
        json.dump(output_dict, fp)


def match_pc_im_laser(split):
    sequence_names = os.listdir(os.path.join(_jrdb_dir, split + "_dataset", "timestamps"))
    for idx, seq_name in enumerate(sequence_names):
        print("({}/{}) Match sensor data for sequence {}".format(
            idx + 1, len(sequence_names), seq_name)
        )
        _match_pc_im_laser_one_sequence(split, seq_name)


if __name__ == "__main__":
    extract_laser_from_rosbag("train")
    match_pc_im_laser("train")
    extract_laser_from_rosbag("test")
    match_pc_im_laser("test")
