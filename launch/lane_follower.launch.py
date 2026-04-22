from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='final_challenge',
            executable='lane_follower',
            name='lane_follower',
            output='screen',
            parameters=[
                {'speed': 4.0},
                {'wheelbase_length': 0.32}
            ]
        )
    ])
