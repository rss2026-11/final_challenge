from setuptools import setup
import os
from glob import glob

package_name = 'final_challenge'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, package_name + '.part_b'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.xml')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'final_challenge'), glob('final_challenge/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Avishai Jeselsohn',
    maintainer_email='avishai@mit.edu',
    description='Final Challenge package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_follower = final_challenge.lane_follower:main',
            'state_machine = final_challenge.part_b.state_machine:main',
            'yolo_detector = final_challenge.part_b.yolo_detector:main',
            'image_saver = final_challenge.part_b.image_saver:main',
            'shell_point_mock = final_challenge.part_b.shell_point_mock:main',
        ],
    },
)
