import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import torch
import numpy as np
import os
import sys
import pathlib
import crownav_base.model.Policy as Policy
# import crownav_base.rl.networks.model.Policy as Policy
import torch
import numpy as np

class CrowdNavPlanner(Node):
    def __init__(self):
        super().__init__('crowdnav_planner')

        # Load CrowdNav model
        self.policy = Policy()
        self.policy.load_model('path_to_trained_model.pth')  # Thay path bằng đường dẫn thực tế

        # ROS2 topics
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)
        self.goal_sub = self.create_subscription(PoseStamped, '/goal_pose', self.plan_path, 10)

    def plan_path(self, goal):
        self.get_logger().info(f'Planning path to goal: {goal.pose.position.x}, {goal.pose.position.y}')
        
        goal_input = torch.tensor([goal.pose.position.x, goal.pose.position.y])
        predicted_path = self.policy.predict(goal_input)

        # Convert model output to ROS2 Path message
        path_msg = Path()
        for point in predicted_path:
            pose = PoseStamped()
            pose.pose.position.x, pose.pose.position.y = point
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)
        self.get_logger().info('Published planned path.')

def main(args=None):
    rclpy.init(args=args)
    node = CrowdNavPlanner()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
