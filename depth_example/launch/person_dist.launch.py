import os

from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    ld = LaunchDescription() #launch파일 생성
    depth_example_pkg = get_package_share_directory('depth_example')

    prarm_dir = LaunchConfiguration(
        'param_dir',
        default=os.path.join(
        depth_example_pkg,
        'param',
        'config.yaml')
        )
    args = [DeclareLaunchArgument(
        'param_dir',
        default_value=prarm_dir
          )]
    
    detect_node = Node(
            package='depth_example',
            executable='detect_node',
            parameters=[prarm_dir],
            output="screen"
    )

    dist_node = Node(
        package='depth_example',
        executable='dist_node',
        parameters=[prarm_dir],
        output='screen'
    )

    # detect_pose_node = Node(
    #     package='depth_example',
    #     executable='detect_pose_node',
    #     parameters=[prarm_dir],
    #     output='screen'
    # )
    
    ld = LaunchDescription(args) #laucnh파일 생성
    ld.add_action(detect_node)
    ld.add_action(dist_node)
    # ld.add_action(detect_pose_node)
    return ld