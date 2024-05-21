import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Int32MultiArray
from cv_bridge import CvBridge
import numpy as np

class DistNode(Node):
    def __init__(self):
        super().__init__('dist_node')

        self.declare_parameter("depth_topic", "/camera/depth/image") # 파라미터를 불러온다.
        self.depth_topic = self.get_parameter('depth_topic').value # 파라미터의 값을 저장한다.


        self.depth_subscriber_ = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_image_callback,
            10
        )

        self.detect_subscriber = self.create_subscription(
            Int32MultiArray,
            '/person_detect',
            self.detect_callback,
            10
        )

        self.dist_publisher = self.create_publisher(
            Float32,
            '/person_dist',
            10
        )

        self.cv_bridge = CvBridge()
        self.depth_image = None

    def depth_image_callback(self, msg):
        self.depth_image = self.cv_bridge.imgmsg_to_cv2(msg,desired_encoding='passthrough')
        self.depth_image = np.array(self.depth_image, dtype=np.float32)
    
    def detect_callback(self, msg):
        tmp = msg.data
        humans = np.reshape(tmp, (-1,6))#배열을 (x,6)로 만들기
    
        if self.depth_image is not None:
            id = 0
            for i in humans:
                x1, y1, x2, y2, conf, cls = i
        

                #if x1 >= 0 and y1 >= 0 and x2 < self.depth_image.shape[1] and y2 < self.depth_image.shape[0]:
                depth_list = [] #박스안의 depth정보 리스트
                id += 1
                for y in range(y1, y2):
                    for x in range(x1, x2):
                        if x >= 0 and x < self.depth_image.shape[1] and y >= 0 and y < self.depth_image.shape[0]:#강의 자료에 없는 한줄 추가 
                            pixel_value = self.depth_image[y, x]
                            depth_list.append(pixel_value)

                if len(depth_list)> 0: # 리스트 길이가 0이면 평균 구할때 분모가 0이된다
                    depth_list.sort()
                    rm_list=int(len(depth_list)/4)
                    depth_list = depth_list[rm_list:rm_list*3] # sort를 하고 노이즈를 제거한다

                    # 평균 계산
                    total = sum(depth_list)
                    length = len(depth_list)
                    average = total / length
                    distance_cm = round(average * 0.1, 4)#cm단위로 변환
                    self.dist_publish(distance_cm, id)#publish함수

    def dist_publish(self, dist, id):
        msg = Float32()
        msg.data = dist
        self.get_logger().info("id:{0} person distance is {1}cm".format(id,
        msg.data))
        self.dist_publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DistNode()
    rclpy.spin(node)
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()