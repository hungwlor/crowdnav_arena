from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'crowdnav_base'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, glob('rl/networks/*.py')),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='CrowdNav Planner for ROS2 Navigation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = crowdnav_base.train:main',
            'test = crowdnav_base.test:main',
            'test_env = crowdnav_base.test_env:main',
        ],
    },
)
