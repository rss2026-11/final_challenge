from setuptools import setup
import os
from glob import glob

package_name = 'final_challenge'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.xml')),
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
            'lane_follower = final_challenge.lane_follower:main'
        ],
    },
)
