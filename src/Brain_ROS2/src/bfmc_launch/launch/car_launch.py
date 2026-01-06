from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        
        # camera node
        Node(
            package='bfmc_hardware',
            executable='camera_node',
            name='camera'
        ),
        
        # serial node
        Node(
            package='bfmc_hardware',
            executable='serial_node',
            name='serial'
        ),
        
        # lane_detection nodes
        Node(
            package='bfmc_perception',
            executable='lane_detector',
            name='lane_detect'
        ),
        
        
        # brain (lane_keeping node)
        Node(
            package='bfmc_brain',
            executable='lane_keeping_node',
            name='lane_keeping'
        )
    ])