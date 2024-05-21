from setuptools import find_packages, setup
import glob, os

package_name = 'depth_example'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob.glob(os.path.join('launch', '*.launch.py'))),
        ('share/' + package_name + '/param', glob.glob(os.path.join('param', '*.yaml')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='seokwon',
    maintainer_email='rgbych@naver.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_node = depth_example.detect_node:main',
            'dist_node = depth_example.dist_node:main',
            'detect_pose_node = depth_example.detect_pose_node:main',
            'detect_seg_node = depth_example.detect_seg_node:main',
            'merge_pose_seg_node = depth_example.merge_pose_seg_node:main',
            'lstm_seg_node = depth_example.lstm_seg_node:main',
        ],
    },
)
