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

risk_score = []
time_to_collision = []
minimum_distance = []
velocity = []

def save_list(file_path, list_to_save):
    with open(file_path, 'wb') as f:
        pickle.dump(list_to_save, f)

def plot_show():
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))

    # risk_score
    axs[0].plot(risk_score, linestyle='-', color='r')
    axs[0].set_title('Risk Score')
    axs[0].set_xlabel('Frame')
    axs[0].set_ylabel('Value')
    axs[0].grid(True)

    # time_to_collision
    axs[1].plot(time_to_collision, linestyle='-', color='b')
    axs[1].set_title('Time to Collision (seconds)')
    axs[1].set_xlabel('Frame')
    axs[1].set_ylabel('Value')
    axs[1].grid(True)

    # minimum_distance
    axs[2].plot(minimum_distance, linestyle='-', color='g')
    axs[2].set_title('Minimum Distance (meters)')
    axs[2].set_xlabel('Frame')
    axs[2].set_ylabel('Value')
    axs[2].grid(True)

    # velocity
    axs[3].plot(velocity, linestyle='-', color='m')
    axs[3].set_title('Velocity (m/s)')
    axs[3].set_xlabel('Frame')
    axs[3].set_ylabel('Value')
    axs[3].grid(True)

    # 레이아웃 조정
    plt.subplots_adjust(hspace=0.8)
    plt.show()
    plot_path = os.path.join(package_path, 'models', '_risk_score_plot.png')
    if os.path.exists(plot_path):
        os.remove(plot_path)
    fig.savefig(plot_path, dpi=400)

class RiskScoreListener(Node):
    def __init__(self):
        super().__init__('risk_score_listener')
        self.risk_score_subscriber = self.create_subscription(
            RiskScoreArray,
            '/risk_estimation/risk_score_array',
            self.risk_score_callback,
            10
        )

    def risk_score_callback(self, msg):
        if len(msg.scores) == 0:
            return
        risk_score.append(msg.scores[0].risk_score)
        time_to_collision.append(msg.scores[0].time_to_collision)
        minimum_distance.append(msg.scores[0].minimum_distance)
        velocity.append(msg.scores[0].velocity)

        if len(risk_score) == 1000:
            plot_show()
            risk_score_path = os.path.join(package_path, 'models', '_risk_score.pkl')
            time_to_collision_path = os.path.join(package_path, 'models', '_time_to_collision.pkl')
            minimum_distance_path = os.path.join(package_path, 'models', '_minimum_distance.pkl')
            velocity_path = os.path.join(package_path, 'models', '_velocity.pkl')
            if os.path.exists(risk_score_path):
                os.remove(risk_score_path)
            if os.path.exists(time_to_collision_path):
                os.remove(time_to_collision_path)
            if os.path.exists(minimum_distance_path):
                os.remove(minimum_distance_path)
            if os.path.exists(velocity_path):
                os.remove(velocity_path)
            save_list(risk_score_path, risk_score)
            save_list(time_to_collision_path, time_to_collision)
            save_list(minimum_distance_path, minimum_distance)
            save_list(velocity_path, velocity)

def main(args=None):
    rclpy.init(args=args)
    node = RiskScoreListener()
    rclpy.spin(node)
    rclpy.shutdown()
if __name__ == '__main__':
    main()

