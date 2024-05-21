import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO
from tensorflow import keras
import torch
import torch.nn as nn



class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size * 2, num_layers=1, batch_first=True)
        self.lstm3 = nn.LSTM(hidden_size * 2, hidden_size * 4, num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(0.1)
        self.lstm4 = nn.LSTM(hidden_size * 4, hidden_size * 2, num_layers=1, batch_first=True)
        self.lstm5 = nn.LSTM(hidden_size * 2, hidden_size, num_layers=1, batch_first=True)
        self.lstm6 = nn.LSTM(hidden_size, hidden_size // 2, num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.lstm7 = nn.LSTM(hidden_size // 2, hidden_size // 4, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size // 4, num_classes)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm1(x, (h0, c0))
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)
        out = self.dropout1(out)
        out, _ = self.lstm4(out)
        out, _ = self.lstm5(out)
        out, _ = self.lstm6(out)
        out = self.dropout2(out)
        out, _ = self.lstm7(out)
        out = self.fc(out[:, -1, :])
        return out



## 하이퍼파라미터 설정
input_size = 34  # 데이터의 차원 (예: x와 y 좌표)
hidden_size = 64  # LSTM의 은닉 상태 크기
num_layers = 2  # LSTM 레이어 수
num_classes = 2  # 분류할 클래스 수 (0 또는 1)


# LSTMModel 생성
loaded_model = LSTMModel(input_size, hidden_size, num_layers, num_classes)

# 불러온 모델의 가중치 로드
loaded_model.load_state_dict(torch.load('lstm_long_data_model_v1.pth'))
loaded_model.eval()



class DetectLSTMSegNode(Node):
    def __init__(self):
        super().__init__('lstm_seg_node')

        self.declare_parameter('rgb_topic', '/camera/color/image')
        self.declare_parameter('conf', 0.5)
        self.declare_parameter('device', 'cpu')

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.conf = self.get_parameter('conf').value
        self.device = self.get_parameter('device').value

        self.colcor_subscriber = self.create_subscription(
            Image,
            self.rgb_topic,
            self.color_image_callback,
            10
        )
        self.pose_publisher = self.create_publisher(
            Int32MultiArray,
            '/lstm_pose_seg',
            10
        )
        self.box_publisher = self.create_publisher(
            Int32MultiArray,
            '/person_box_tracking',
            10
        )
        self.seg_publisher = self.create_publisher(
            Int32MultiArray,
            '/person_seg_tracking',
            10
        )
        self.cv_bridge = CvBridge()

        self.model = YOLO('yolov8n-pose.pt')
        self.model2 = YOLO('yolov8n.pt')
        self.model3 = YOLO('yolov8n-seg.pt')

        # self.visualization = True


    def color_image_callback(self, msg):
        color_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        results = self.model.predict(color_image, conf=self.conf, device=self.device) # detect
        # result = results[0].boxes.data.cpu().numpy()
        # keypoints = result.keypoints
        model2_results = self.model2.predict(color_image, conf=self.conf, device=self.device)
        model2_result = model2_results[0].boxes.data.cpu().numpy()
        humans = model2_result[model2_result[:, -1] == 0]

        person_tracking = self.model3.track(color_image, persist=True, conf=0.5, classes=0, device="cuda", max_det=1)[0]
        tracking_box = []
        tracking_seg = []
        frame_keypoints_xyn = []


        for result in results:
            for idx, keypoints in enumerate(result.keypoints):
                try:
                    xyn = keypoints.xyn.cpu().numpy()
                    xyn_list = xyn.tolist()
                    flattened_xyn = [point for sublist in xyn_list[0] for point in sublist]
                    frame_keypoints_xyn.append(flattened_xyn)

                    if xyn.size == 0:
                        continue

   
                    # LSTM 모델을 사용하여 예측
                    if len(frame_keypoints_xyn):
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
                                print("LSTM 모델 예측 결과 (클래스 확률):", class_probabilities_numpy)
                                if (class_probabilities_numpy[0][1] < 0.6).any():
                                    msg = Int32MultiArray()
                                    msg.data = [0]
                                    self.box_publisher.publish(msg)
                                else:
                                    msg = Int32MultiArray()
                                    msg.data = [1]
                                    self.seg_publisher.publish(msg)



                                    id = 0
                                    for row in humans :
                                        id += 1
                                        x1, y1, x2, y2, conf, cls = row
                                        cx = int(x1 + x2) // 2
                                        cy = int(y1 + y2) // 2
                                        center = (cx, cy)

                                        cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255, 0), 2)
                                        cv2.circle(color_image, center, 3, (0, 0, 255), -1)
                                        cv2.putText(color_image, "person " + str(id), (int(x1), int(y1)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                                        if person_tracking.boxes.id==None:
                                            return
                                        dst = color_image[:, :, 2].copy()
                                        cv2.addWeighted(color_image[:, :, 2], 0.5, np.logical_or.reduce(person_tracking.masks.data.cpu().numpy(), axis=0).astype(np.uint8) * 255, 0.5, 0, dst)
                                        color_image[:, :, 2] = dst
                                        for idx in range(len(person_tracking.boxes)):
                                            x1, y1, x2, y2 = person_tracking.boxes[idx].data.tolist()[0][:4]
                                            x1, y1, x2, y2 = map(round, [x1, y1, x2, y2])

                                            tracking_box.extend([int(person_tracking.boxes[idx].id[0]), x1, y1, x2, y2])

                                            tracking_seg.extend([1] + person_tracking.masks.data[idx].cpu().numpy().astype(np.uint8).flatten().tolist())

                                            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255), 4)
                                            # 텍스트를 박스 위에 추가 (배경을 빨간색으로, 글자를 흰색으로)
                                            text = "ID: " + str(int(person_tracking.boxes[idx].id[0])) + ", Acc: " + str(float(person_tracking.boxes[idx].data[0][-2]))[:4]
                                            font = cv2.FONT_HERSHEY_SIMPLEX
                                            font_scale = 1
                                            font_thickness = 2
                                            text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                                            text_x = x1 - 2  # 텍스트를 그릴 x 좌표
                                            text_y = y1 + 22  # 텍스트를 그릴 y 좌표 (박스 위에 위치)
                                            # 텍스트 배경 박스를 그리기
                                            cv2.rectangle(color_image, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), (0, 0, 255), cv2.FILLED)
                                            # 텍스트를 흰색으로 그리기
                                            cv2.putText(color_image, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

                                        # if self.visualization:
                                        #     cv2.imshow("YOLOv8 Tracking", color_image)
                                        #     cv2.waitKey(1)
                                        if len(tracking_box) > 0:
                                            msg = Int32MultiArray()
                                            msg.data = tracking_box
                                            self.box_publisher.publish(msg)
                                        if len(tracking_seg) > 0:
                                            msg = Int32MultiArray()
                                            msg.data = tracking_seg
                                            self.seg_publisher.publish(msg)






                except IndexError as e :
                    continue







        cv2.imshow("YOLOv8 Skeleton keypoint", color_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DetectLSTMSegNode()
    rclpy.spin(node)
    rclpy.shutdown()
if __name__ == '__main__':
    main()

