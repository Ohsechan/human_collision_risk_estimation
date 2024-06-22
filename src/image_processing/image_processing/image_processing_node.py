import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO
from hcre_msgs.msg import PoseTracking, PoseTrackingArray

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from ament_index_python.packages import get_package_prefix

def find_package_path(package_name):
    package_path = get_package_prefix(package_name)
    package_path = os.path.dirname(package_path)
    package_path = os.path.dirname(package_path)
    package_path = os.path.join(package_path, "src", package_name)
    return package_path

package_path = find_package_path('image_processing')

frame_keypoints_xyn = [[]] # id별로 keypoints 저장하는 공간 (id, stack, keypoints)

processtime = []

class LSTMModel(nn.Module):
    def __init__(self, num_features = 34):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(num_features, 64, batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)

        self.fc1 = nn.Linear(64, 256)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(256, 64)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(64, 16)
        self.batch_norm4 = nn.BatchNorm1d(16)
        self.dropout4 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(16, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last output of LSTM

        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = F.relu(self.fc1(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        x = F.relu(self.fc2(x))
        x = self.batch_norm3(x)
        x = self.dropout3(x)

        x = F.relu(self.fc3(x))
        x = self.batch_norm4(x)
        x = self.dropout4(x)

        x = torch.sigmoid(self.fc4(x))
        return x

class ImageProcessingNode(Node):
    def __init__(self):
        super().__init__('image_processing_node')
        self.colcor_subscriber = self.create_subscription(
            Image,
            '/AMR/D435i/color/image_raw',
            self.color_image_callback,
            10
        )
        self.pose_tracking_data_publisher = self.create_publisher(
            PoseTrackingArray,
            '/image_processing/pose_tracking',
            10
        )
        self.cv_bridge = CvBridge()
        self.model = YOLO(os.path.join(package_path,'models','yolov8n-pose.pt'))
        self.lstm_model = LSTMModel()
        lstm_model_path = os.path.join(package_path,'models','lstm_model.pth')
        self.lstm_model.load_state_dict(torch.load(lstm_model_path))

    def calcualte_processtime(self, now_time):
        if (len(processtime) <= 1050):
            processtime.append(self.get_clock().now().nanoseconds - now_time)
        if (len(processtime) == 1050):
            data_array = np.array(processtime[50:])
            mean = np.mean(data_array)
            median = np.median(data_array)
            std_dev = np.std(data_array)
            min_value = np.min(data_array)
            max_value = np.max(data_array)
            print(f'Mean: {mean}')
            print(f'Median: {median}')
            print(f'Standard Deviation: {std_dev}')
            print(f'Min: {min_value}')
            print(f'Max: {max_value}')

    def color_image_callback(self, msg):
        color_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model.track(color_image,
                    persist=True, # persisting tracks between frames
                    verbose=False, # no print
                    device="cuda:0")
        
        # yolo 적용 여부 확인용
        # cv2.imshow("YOLOv8 Tracking", results[0].plot()) # show result
        # cv2.waitKey(1)

        pose_tracking_data = PoseTrackingArray()
        pose_tracking_data.timestamp = (msg.header.stamp.sec % 1000000) * 1000 + msg.header.stamp.nanosec // 1000000
        id_keypionts = [] # to publish keypoints
        for person in results[0]: # for every person
            # now_time = self.get_clock().now().nanoseconds # todo: 실행시간 계산용
            each_pose = PoseTracking()

            keypoints = person.keypoints
            xyn = keypoints.xyn.cpu()
            xyn_list = xyn.tolist()
            flattened_xyn = [point for sublist in xyn_list[0] for point in sublist]

            if xyn.size == 0:
                continue

            # 예외처리 : id 없는 경우
            if (person.boxes.id == None):
                continue
            id = int(person.boxes.id.item()) # id 값 받아오기
            # id에 해당하는 저장공간 확보
            while(len(frame_keypoints_xyn) < id):
                frame_keypoints_xyn.append([])
            # 각 id마다 길이가 5가 넘지 않도록 과거 데이터 삭제 후 현재 데이터 추가
            if len(frame_keypoints_xyn[id-1]) == 5:
                frame_keypoints_xyn[id-1] = frame_keypoints_xyn[id-1][1:]
            frame_keypoints_xyn[id-1].append(flattened_xyn)

            # id와 keypoints를 each_pose에 저장
            each_pose.id = id
            x_list, y_list = [], []
            for sublist in keypoints.xy.cpu().tolist()[0]:
                x_list.append(int(sublist[0]))
                y_list.append(int(sublist[1]))
            each_pose.x = x_list
            each_pose.y = y_list
            each_pose.pose = True # stand 라고 가정

            # LSTM 모델을 사용하여 예측
            if len(frame_keypoints_xyn[id-1]) > 0:
                input_data = torch.tensor(frame_keypoints_xyn[id-1])
                input_data = input_data.unsqueeze(0) # (1, sequence_length, input_size)
                self.lstm_model.eval()
                y_pred_prob = self.lstm_model(input_data)
                y_pred = (y_pred_prob.item() > 0.5)
                
                if y_pred == 0:
                    each_pose.pose = False # sit

            pose_tracking_data.posetracking.append(each_pose)
            # self.calcualte_processtime(now_time) # todo: 실행시간 계산용
        '''
        pose tracking format description
            time stamp (millisecond),
            id, pose, keypoints.xy,
            id, pose, keypoints.xy,
            id, pose, keypoints.xy,
            ...
        '''
        self.pose_tracking_data_publisher.publish(pose_tracking_data)

def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessingNode()
    rclpy.spin(node)
    rclpy.shutdown()
if __name__ == '__main__':
    main()

