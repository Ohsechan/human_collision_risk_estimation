import rclpy
from rclpy.node import Node
from ultralytics import YOLO
from hcre_msgs.msg import RiskScore, RiskScoreArray
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from ament_index_python.packages import get_package_prefix

def find_package_path(package_name):
    package_path = get_package_prefix(package_name)
    package_path = os.path.dirname(package_path)
    package_path = os.path.dirname(package_path)
    package_path = os.path.join(package_path, "src", package_name)
    return package_path

package_path = find_package_path('image_processing')

def load_list(file_path):
    with open(file_path, 'rb') as f:
        loaded_list = pickle.load(f)
    return loaded_list

risk_score = load_list(os.path.join(package_path, 'models', '_risk_score.pkl'))
time_to_collision = load_list(os.path.join(package_path, 'models', '_time_to_collision.pkl'))
minimum_distance = load_list(os.path.join(package_path, 'models', '_minimum_distance.pkl'))
velocity = load_list(os.path.join(package_path, 'models', '_velocity.pkl'))

def plot_show():
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

    # risk_score
    axs[0].plot(risk_score, linestyle='-', color='r')
    axs[0].set_title('Risk Score')
    axs[0].set_xlabel('Index')
    axs[0].set_ylabel('Value')
    axs[0].grid(True)

    # time_to_collision
    axs[1].plot(time_to_collision, linestyle='-', color='b')
    axs[1].set_title('Time to Collision (seconds)')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Value')
    axs[1].grid(True)

    # minimum_distance
    axs[2].plot(minimum_distance, linestyle='-', color='g')
    axs[2].set_title('Minimum Distance (meters)')
    axs[1].set_xlabel('Index')
    axs[1].set_ylabel('Value')
    axs[2].grid(True)

    # velocity
    axs[3].plot(velocity, linestyle='-', color='m')
    axs[3].set_title('Velocity (m/s)')
    axs[3].set_xlabel('Index')
    axs[3].set_ylabel('Value')
    axs[3].grid(True)

    # 레이아웃 조정
    plt.subplots_adjust(hspace=0.5)
    plt.show()
    fig.savefig(os.path.join(package_path, 'models', '_risk_score_plot.png'), dpi=400)

def main(args=None):
    plot_show()

if __name__ == '__main__':
    main()

