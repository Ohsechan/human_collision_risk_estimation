import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn

class DetectPoseNode(Node):
    def __init__(self):
        super().__init__('detect_pose_node')
        self.colcor_subscriber = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.color_image_callback,
            10
        )
        self.pose_publisher = self.create_publisher(
            Int32MultiArray,
            '/person_pose_acting',
            10
        )
        self.cv_bridge = CvBridge()
        self.model = YOLO('yolov8n-pose.pt')

    def calculate_angle(self, joint1, joint2, joint3): 
        # 어깨 골반 무릎 순서
        # 하지만 사실 점 3개를 해도 된다
        # 5,6 어깨
        # 11, 12 골반
        # 13, 14 무릎
        # 각도 계산
        vector1 = np.squeeze(np.array(joint1.cpu())) - np.squeeze(np.array(joint2.cpu()))
        vector2 = np.squeeze(np.array(joint3.cpu())) - np.squeeze(np.array(joint2.cpu()))

        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)

        cosine_theta = dot_product / (magnitude1 * magnitude2)
        angle = np.arccos(np.clip(cosine_theta, -1.0, 1.0))

        return np.degrees(angle)



    def color_image_callback(self, msg):
        color_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model.track(color_image,
                    persist=True, # persisting tracks between frames
                    verbose=False, # no print
                    imgsz=(480,640), device="cuda:0")
        cv2.imshow("YOLOv8 Tracking", results[0].plot()) # show result
        cv2.waitKey(1)
        for i in results[0].keypoints.xy:
            print(i)
        print()

        frame_keypoints_xyn = []
        for person in results[0]:
            try:
                keypoints = person.keypoints
                xyn = keypoints.xyn.cpu().numpy()
                xyn_list = xyn.tolist()
                flattened_xyn = [point for sublist in xyn_list[0] for point in sublist]
                frame_keypoints_xyn.append(flattened_xyn)

                if xyn.size == 0:
                    continue

                # LSTM 모델을 사용하여 예측
                if len(frame_keypoints_xyn) > 0:
                    new_data = np.array(frame_keypoints_xyn)  # 예측하고자 하는 새로운 데이터

                    # 3-D 텐서로 입력 데이터 조정
                    new_data = torch.FloatTensor(new_data)  # (sequence_length, input_size)
                    new_data = new_data.unsqueeze(0)  # (1, sequence_length, input_size)

                    if new_data.size(-1) == input_size: # 입력 데이터의 차원이 맞는지 확인
                        lstm_prediction = loaded_model(new_data)

                        # 예측 결과를 클래스 확률로 변환
                        softmax = torch.nn.Softmax(dim=1)
                        class_probabilities = softmax(lstm_prediction)
                        class_probabilities_numpy = class_probabilities.cpu().detach().numpy()

                        # 예측 결과 출력
                        # print("LSTM 모델 예측 결과 (클래스 확률):", class_probabilities_numpy)
                        if (class_probabilities_numpy[0][1] < 0.6).any():
                            msg = Int32MultiArray()
                            msg.data = [0] # 'sit'
                            self.box_publisher.publish(msg)
                            print("sit")
                        else:
                            msg = Int32MultiArray()
                            msg.data = [1] # 'stand'
                            self.seg_publisher.publish(msg)
                            print("stand")

            except IndexError as e :
                continue

        # for result in results:
        #     for idx, keypoints in enumerate(result.keypoints):
        #         try:
        #             xyn = keypoints.xyn
        #             if xyn.size == 0:
        #                 continue

        #             # selected_keypoints = xyn[0, [5, 11, 13], :]
        #             shoulder = xyn[0, [5], :] # joint1
        #             hip = xyn[0, [11], :]     # joint2
        #             knee = xyn[0, [13], :]    # joint3

        #             angle = self.calculate_angle(shoulder, hip, knee)

        #             threshold_angle = 120
                    
        #             if angle < threshold_angle:
        #                 msg = Int32MultiArray()
        #                 msg.data = [1] # 'sit'
        #                 self.pose_publisher.publish(msg)
        #                 print("sit")
        #             else:
        #                 msg = Int32MultiArray()
        #                 msg.data = [2] # 'stand'
        #                 self.pose_publisher.publish(msg)
        #                 print("stand")

        #         except IndexError as e :
        #             continue

def main(args=None):
    rclpy.init(args=args)
    node = DetectPoseNode()
    rclpy.spin(node)
    rclpy.shutdown()
if __name__ == '__main__':
    main()