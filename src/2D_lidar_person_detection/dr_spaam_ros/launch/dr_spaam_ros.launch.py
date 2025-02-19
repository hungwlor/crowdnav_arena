#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
import launch
import launch_ros.actions

def generate_launch_description():
    pkg_share = get_package_share_directory('dr_spaam_ros')
    return launch.LaunchDescription([
        launch_ros.actions.Node(
            package='dr_spaam_ros',
            executable='node.py',  # or simply 'node' if renamed in your package
            name='dr_spaam_ros',
            output='screen',
            parameters=[
                os.path.join(pkg_share, 'config', 'dr_spaam_ros.yaml'),
                os.path.join(pkg_share, 'config', 'topics.yaml')
            ]
        )
    ])
