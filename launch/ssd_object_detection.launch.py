import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    ssd_params = os.path.join(
        get_package_share_directory('ssd_object_detection'),
        'params',
        'config.yaml'
        )
    
    ssd_node = Node(
            package='ssd_object_detection',
            executable='ssd_object_detection_node',
            output='screen',
            name='ssd_node',
            parameters=[ssd_params])
    
    ld.add_action(ssd_node)
    return ld