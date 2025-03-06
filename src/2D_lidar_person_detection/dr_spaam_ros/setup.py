from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dr_spaam_ros'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description=' for ROS2 Navigation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector=dr_spaam_ros.node:main',
        ],
    },
)
