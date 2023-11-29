import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from cv_bridge import CvBridge
import numpy as np
import cv2

from ultralytics import YOLO

class DetectNode(Node):
    def __init__(self):
        super().__init__('detect_node')

        self.declare_parameter('rgb_topic', '/camera/color/image')
        self.declare_parameter('conf', 0.5)
        self.declare_parameter("device", "cpu") #파라미터를 불러온다

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.conf = self.get_parameter('conf').value
        self.device = self.get_parameter('device').value # 파라미터의 값을 저장한다.

        self.colcor_subscriber = self.create_subscription(
            Image,
            self.rgb_topic,
            self.color_image_callback,
            10
        )
    
        self.detect_publisher = self.create_publisher(
            Int32MultiArray,
            '/person_detect',
            10
        )

        self.cv_bridge = CvBridge()
    
        self.model = YOLO('yolov8n.pt') #욜로 모델 불러오기
    
    def color_image_callback(self, msg):
        color_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        results = self.model.predict(color_image, conf=self.conf, device=self.device) #detect
        result = results[0].boxes.data.cpu().numpy()
        humans = result[result[:,-1] == 0]

        if len(humans) > 0:
            tmp = np.reshape(humans, -1) #1차원 array로 만들기
            msg = Int32MultiArray()
            tmp = tmp.astype(int).tolist()
            msg.data = tmp
            self.detect_publisher.publish(msg) #/person_detect으로 퍼블리시
        
        id = 0
        for row in humans:
            id += 1
            x1, y1, x2, y2, conf, cls = row
            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2
            center = (cx, cy)

            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0,255, 0), 1)
            cv2.circle(color_image, center, 3, (0, 0, 255), -1)
            cv2.putText(color_image, "person " + str(id), (int(x1), int(y1)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("image", color_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DetectNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
