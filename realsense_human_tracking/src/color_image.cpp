#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/highgui/highgui.hpp"

class ImageSubscriber : public rclcpp::Node {
    cv::Mat frame;
public:
    ImageSubscriber() : Node("image_subscriber_node") {
        // Create a subscriber for the /camera/color/image_raw topic
        subscription_ = create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw", 1, std::bind(&ImageSubscriber::colorImageCallback, this, std::placeholders::_1));

        // Initialize OpenCV window for displaying the image
        cv::namedWindow("Color Image", cv::WINDOW_AUTOSIZE);
    }

private:
    void colorImageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {

        frame.create(msg->height, msg->width, CV_8UC3);
        frame.data = const_cast<unsigned char *>(msg->data.data());
        // frame.step = msg->step;
        cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);

        int x = 100;
        int y = 50;
        cv::Vec3b pixel = frame.at<cv::Vec3b>(y, x);
        int blue = pixel[0];
        int green = pixel[1];
        int red = pixel[2];
        std::cout << "Pixel at (" << x << ", " << y << "): " << "B=" << blue << ", G=" << green << ", R=" << red << std::endl;

        cv::imshow("Received Image", frame);
        if(cv::waitKey(10) == 27 ) exit(1);
    }

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    cv::Mat color_image_;
};

int main(int argc, char** argv) {
    // Initialize ROS 2
    rclcpp::init(argc, argv);

    // Create an instance of the ImageSubscriber class
    auto node = std::make_shared<ImageSubscriber>();

    // Spin ROS 2
    rclcpp::spin(node);

    // Shutdown ROS 2
    rclcpp::shutdown();

    return 0;
}
