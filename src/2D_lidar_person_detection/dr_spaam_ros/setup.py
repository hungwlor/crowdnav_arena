from setuptools import setup
import os
from glob import glob
package_name = 'dr_spaam_ros'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
    (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    package_dir={'': 'src'},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dan Jia',
    maintainer_email='jia@vision.rwth-aachen.de',
    description='ROS interface for DR-SPAAM detector',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # For example, if you have a node executable:
            'dr_spaam_ros_node = dr_spaam_ros.node:main'
        ],
    },
)
