import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import numpy as np
import cv2
from ultralytics import YOLO

class DetectPoseSegNode(Node):
    def __init__(self):
        super().__init__('merge_pose_seg_node')

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
            '/merge_pose_seg',
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
        
        results = self.model.predict(color_image, conf=self.conf, device=self.device) # detect
        # result = results[0].boxes.data.cpu().numpy()
        # keypoints = result.keypoints
        model2_results = self.model2.predict(color_image, conf=self.conf, device=self.device)
        model2_result = model2_results[0].boxes.data.cpu().numpy()
        humans = model2_result[model2_result[:, -1] == 0]

        person_tracking = self.model3.track(color_image, persist=True, conf=0.5, classes=0, device="cuda", max_det=1)[0]
        tracking_box = []
        tracking_seg = []


        for result in results:
            for idx, keypoints in enumerate(result.keypoints):
                try:
                    xyn = keypoints.xyn
                    if xyn.size == 0:
                        continue

                    # selected_keypoints = xyn[0, [5, 11, 13], :]
                    shoulder = xyn[0, [5], :] # joint1
                    hip = xyn[0, [11], :]     # joint2
                    knee = xyn[0, [13], :]    # joint3



                    angle = self.calculate_angle(shoulder, hip, knee)

                    threshold_angle = 120
                    
                    if angle < threshold_angle:
                        msg = Int32MultiArray()
                        msg.data = [1] # 'sit'
                        self.pose_publisher.publish(msg)
                    else:
                        msg = Int32MultiArray()
                        msg.data = [2] # 'stand'
                        self.pose_publisher.publish(msg)

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
    node = DetectPoseSegNode()
    rclpy.spin(node)
    rclpy.shutdown()
if __name__ == '__main__':
    main()

