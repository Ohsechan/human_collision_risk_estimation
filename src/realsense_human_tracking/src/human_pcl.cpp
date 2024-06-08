#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/int32_multi_array.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include <cmath>
#include <iostream>
#include <thread>
#include <deque>
#include "rclcpp/clock.hpp"
#include <fstream>

// #include <realsense_human_tracking/msg/risk_score.hpp>
// #include "realsense_human_tracking/msg/risk_score_array.hpp"

#include <pcl/common/angles.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <iomanip> // for setw, setfill

// Parameters
const int OLD_DATA_REFERENCE_TIME = 1000; // Unit: ms
const int COLLISION_REFERENCE_TIME = 10000; // Unit: ms
const int COLLISION_REFERENCE_DISTANCE = 1000; // Unit: mm

// Realsesne depth camera의 내부 파라미터 구조
struct rs2_intrinsics {
    float         ppx;       /**< Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge */
    float         ppy;       /**< Vertical coordinate of the principal point of the image, as a pixel offset from the top edge */
    float         fx;        /**< Focal length of the image plane, as a multiple of pixel width */
    float         fy;        /**< Focal length of the image plane, as a multiple of pixel height */
    float         coeffs[5]; /**< Distortion coefficients */
};

// location 및 timestamp
struct Point {
    float x=0, y=0, z=0;
    int time=0;
};

// 속도
struct Velocity {
    float vx=0, vy=0, vz=0;
    int time=0;
};

// Risk score
struct RiskScore {
    int id=0;
    float risk_score = 0, collision_time=0, distance=0, velocity = 0;
};

// 가속도
struct Acceleration {
    float ax=0, ay=0, az=0;
};

// PCL 뷰터 초기화 함수
pcl::visualization::PCLVisualizer::Ptr rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud){
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

// 통계적인 방법으로 Outlier 제거
void removeOutliers(cv::Mat& mat) {
    // 평균과 표준 편차 계산
    cv::Scalar mean, stddev;
    cv::Mat nonZeroMask = (mat != 0);
    cv::meanStdDev(mat, mean, stddev, nonZeroMask);

    float now_std = stddev[0];
    float threshold = 33.0;// 10 / pow(mean[0], 1.0 / 5.0);
    
    // for (int i = 0; i < iteration; i++) {
    while(true) {
        nonZeroMask = (mat != 0);
        cv::meanStdDev(mat, mean, stddev, nonZeroMask);

        if (now_std - stddev[0] <= now_std * 0.0001){
            threshold *= 0.9;
        }

        now_std = stddev[0];

        // Z-score를 계산하고, 임계값 이상인 값을 0으로 설정
        mat.forEach<float>([&](float& pixel, const int* position) -> void {
            double zScore = (pixel - mean[0]) / stddev[0];
            if (std::abs(zScore) > threshold) {
                pixel = 0.0;
            }
        });
        if (now_std < std::pow(mean[0], 1.27) / 11.6){
            break;
        }
    }
    nonZeroMask = (mat != 0);
    cv::meanStdDev(mat, mean, stddev, nonZeroMask);
    // printf("mean : %f, stddev : %f, goal : %f, thres : %f\n", mean[0], stddev[0], std::pow(mean[0], 1.27) / 10.5, threshold);
}

// 경계선 부분 노이즈 제거 함수
void removeOutline(cv::Mat& mat) {
    // mat 크기
    cv::Mat clonedMat = mat.clone();
    int rows = clonedMat.rows;
    int cols = clonedMat.cols;

    // 2x2 크기의 커널
    int kernelSize = 2;
    cv::Mat kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_8U);

    for (int i = 0; i < rows - kernelSize; i += 2) {
        for (int j = 0; j < cols - kernelSize; j += 2) {
            cv::Mat regionOfInterest = clonedMat(cv::Rect(j, i, kernelSize, kernelSize));
            // 커널 안에 0이 하나라도 있으면 해당 커널의 값을 모두 0으로 만듦
            if (cv::countNonZero(regionOfInterest <= 0) > 0) {
                kernel.copyTo(mat(cv::Rect(j, i, kernelSize, kernelSize)));
            }
        }
    }
}

// segmentation mask 부분의 depth data만 필터링하는 함수
cv::Mat human_filter(cv::Mat depth_image_, cv::Mat mat_human){
    cv::Mat result;
    mat_human.convertTo(mat_human, CV_32F);
    cv::multiply(depth_image_, mat_human, result);
    return result;
}

// pixel과 depth 정보를 실제 세계의 3차원 직교좌표계로 변환하는 함수
static void rs2_deproject_pixel_to_point(pcl::PointXYZRGB& point, const struct rs2_intrinsics * intrin, int pixel_x, int pixel_y, float depth) {
    float x = (pixel_x - intrin->ppx) / intrin->fx;
    float y = (pixel_y - intrin->ppy) / intrin->fy;
    // if Distortion Model == Brown Conrady
    float r2  = x*x + y*y;
    float f = 1 + intrin->coeffs[0]*r2 + intrin->coeffs[1]*r2*r2 + intrin->coeffs[4]*r2*r2*r2;
    float ux = x*f + 2*intrin->coeffs[2]*x*y + intrin->coeffs[3]*(r2 + 2*x*x);
    float uy = y*f + 2*intrin->coeffs[3]*x*y + intrin->coeffs[2]*(r2 + 2*y*y);
    x = ux;
    y = uy;
    point.y = depth * x;
    point.z = depth * y;
    point.x = depth;
    point.r = 255;
    point.g = std::round(255 - 255 * depth / 5);
    point.b = std::round(255 - 255 * depth / 5);
}

// point cloud의 평균 위치를 구하는 함수
Point calculateAverage(const pcl::PointCloud<pcl::PointXYZRGB> pointClouds, int millisec) {
    Point result;
    for (const auto& point : pointClouds.points) {
        result.x += point.x;
        result.y += point.y;
        result.z += point.z;
    }
    result.x /= pointClouds.size();
    result.y /= pointClouds.size();
    result.z /= pointClouds.size();
    result.time = millisec;
    // printf("%f, %f, %f, %f\n", result.x, result.y, result.z, result.time);
    return result;
}

// vector의 평균을 계산하는 함수
float calculateAverage(const std::vector<float>& data) {
    if (data.empty()) {
        return 0.0; // 빈 벡터의 경우 0을 반환하거나 예외 처리를 수행할 수 있습니다.
    }

    // accumulate 함수를 사용하여 합을 계산하고, 크기로 나누어 평균을 얻습니다.
    return std::accumulate(data.begin(), data.end(), 0.0) / data.size();
}

// 평균 속도를 계산하는 함수
Velocity calculateVelocity(const Point& p1, const Point& p2) {
    Velocity v;
    v.vx = (p2.x - p1.x) / (p2.time - p1.time) * 1000; // meterpersec
    v.vy = (p2.y - p1.y) / (p2.time - p1.time) * 1000;
    v.vz = (p2.z - p1.z) / (p2.time - p1.time) * 1000;
    return v;
}

// 평균 가속도를 계산하는 함수
Acceleration calculateAcceleration(const Velocity& v1, const Velocity& v2, float time) {
    Acceleration a;
    a.ax = (v2.vx - v1.vx) / time * 1000; // meterpersec^2
    a.ay = (v2.vy - v1.vy) / time * 1000;
    a.az = (v2.vz - v1.vz) / time * 1000;
    return a;
}

// Node 정의
class DistNode : public rclcpp::Node {
    public:
        // publisher 및 subscriber 정의
        DistNode() : Node("dist_node") {
            camera_info_subscriber_ = create_subscription<sensor_msgs::msg::CameraInfo>(
                "/camera/depth/camera_info",
                10,
                std::bind(&DistNode::cameraInfoCallback, this, std::placeholders::_1)
            );
            depth_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/camera/aligned_depth_to_color/image_raw",
                10,
                std::bind(&DistNode::depthImageCallback, this, std::placeholders::_1)
            );
            color_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/camera/color/image_raw",
                10,
                std::bind(&DistNode::colorImageCallback, this, std::placeholders::_1)
            );
            seg_subscription_ = this->create_subscription<std_msgs::msg::Int32MultiArray>(
                "/person_seg_tracking",
                10,
                std::bind(&DistNode::segCallback, this, std::placeholders::_1)
            );
            xyzt_publisher_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/xyzt_meter_millisec", 10);
        }

    private:
        // 클래스에서 사용할 변수 정의
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscriber_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subscription_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_subscription_;
        rclcpp::Subscription<std_msgs::msg::Int32MultiArray>::SharedPtr seg_subscription_;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr xyzt_publisher_;
        cv_bridge::CvImagePtr cv_ptr;
        cv::Mat color_image_;
        cv::Mat depth_image_;
        std::deque<cv::Mat> depth_image_deque;
        std::deque<int> depth_image_time_stamp_deque;
        std::deque<Point> xyzt_deque;
        std::vector<RiskScore> risk_list;
        std::vector<std::vector<std::deque<Point>>> id_keypoint_xyzt_vector;
        std::vector<std::vector<std::deque<Velocity>>> id_keypoint_velocity_vector;
        rs2_intrinsics intrinsics;
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
        pcl::visualization::PCLVisualizer::Ptr viewer = rgbVis(point_cloud_ptr);
        int now_time = 0;
        std::vector<float> flight_time, cycle_time;
        int now_time2 = 0;
        std::vector<int> processTime; // for calculate process time

        void calc_processtime(int now_time_input) {
            processTime.push_back((this->now().nanoseconds() / 1000000) % 1000000000 - now_time_input);
            if (processTime.size() == 1000){
                const std::string filename = "processTime.txt";
                std::ofstream outFile(filename);
                if (!outFile.is_open()) {
                    std::cerr << "파일을 열 수 없습니다." << std::endl;
                    return;
                }
                for (const auto& value : processTime) {
                    outFile << value << std::endl;
                }
                outFile.close();
                printf("%dms\n", (this->now().nanoseconds() / 1000000) % 1000000000 - now_time_input);
            }
        }

        int return_millisecond(int sec, int nanosec){
            return (sec % 1000000) * 1000 + nanosec / 1000000; // 9자리 millisecond 반환
        }

        // pixel과 depth 정보를 실제 세계의 3차원 직교좌표계로 변환하는 함수
        Point rs2_deproject_pixel_to_point_modified(const struct rs2_intrinsics * intrin, int pixel_x, int pixel_y, float depth, int timestamp) {
            float x = (pixel_x - intrin->ppx) / intrin->fx;
            float y = (pixel_y - intrin->ppy) / intrin->fy;
            // if Distortion Model == Brown Conrady
            float r2  = x*x + y*y;
            float f = 1 + intrin->coeffs[0]*r2 + intrin->coeffs[1]*r2*r2 + intrin->coeffs[4]*r2*r2*r2;
            float ux = x*f + 2*intrin->coeffs[2]*x*y + intrin->coeffs[3]*(r2 + 2*x*x);
            float uy = y*f + 2*intrin->coeffs[3]*x*y + intrin->coeffs[2]*(r2 + 2*y*y);
            x = ux;
            y = uy;
            Point point;
            point.x = x * depth;
            point.y = y * depth;
            point.z = depth;
            point.time = timestamp;
            return point;
        }

        Velocity calculate_velocity(const Point p1, const Point p2) {
            Velocity vel;
            int deltaTime = p2.time - p1.time;
            if (deltaTime == 0) {
                std::cerr << "Error: The time difference between points is zero." << std::endl;
                return vel; // Return default velocity
            }
            vel.vx = (p2.x - p1.x) / deltaTime;
            vel.vy = (p2.y - p1.y) / deltaTime;
            vel.vz = (p2.z - p1.z) / deltaTime;
            vel.time = p2.time;
            return vel;
        }

        Point calculate_average_point(const std::vector<std::deque<Point>> point_vector) {
            Point average;
            for (const auto& points : point_vector) {
                for (const auto& point : points) {
                    average.x += point.x;
                    average.y += point.y;
                    average.z += point.z;
                    average.time++;
                }
            }

            average.x /= average.time;
            average.y /= average.time;
            average.z /= average.time;

            return average;
        }

        Velocity calculate_average_velocity(const std::vector<std::deque<Velocity>> velocity_vector) {
            Velocity average;
            for (const auto& velocities : velocity_vector) {
                for (const auto& velocity : velocities) {
                    average.vx += velocity.vx;
                    average.vy += velocity.vy;
                    average.vz += velocity.vz;
                    average.time++;
                }
            }

            average.vx /= average.time;
            average.vy /= average.time;
            average.vz /= average.time;

            return average;
        }

        std::vector<float> find_closest_time_distance(const Point loc, const Velocity vel) {
            // Calculate the dot product of loc and vel
            float dot_product = loc.x * vel.vx + loc.y * vel.vy + loc.z * vel.vz;
            
            // Calculate the magnitude squared of the velocity vector
            float vel_magnitude_squared = vel.vx * vel.vx + vel.vy * vel.vy + vel.vz * vel.vz;
            
            // Calculate the parameter t that minimizes the distance
            float t = -dot_product / vel_magnitude_squared;
            
            // Calculate the closest point on the line
            float closest_x = loc.x + t * vel.vx;
            float closest_y = loc.y + t * vel.vy;
            float closest_z = loc.z + t * vel.vz;
            
            // Calculate the distance from (0, 0, 0) to the closest point
            float distance = std::sqrt(closest_x * closest_x + closest_y * closest_y + closest_z * closest_z);
            
            // Return the closest time and the distance
            return {t, distance};
        }

        // color image를 color_image_ 변수에 저장하는 함수
        void colorImageCallback(const sensor_msgs::msg::Image::ConstPtr msg) {
            color_image_.create(msg->height, msg->width, CV_8UC3);
            color_image_.data = const_cast<unsigned char *>(msg->data.data());
            color_image_.step = msg->step;
            cv::cvtColor(color_image_, color_image_, cv::COLOR_RGB2BGR);
        }

        // camera의 intrinsic parameter를 받아오는 함수
        void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
            intrinsics.ppx = msg->k[2];
            intrinsics.ppy = msg->k[5];
            intrinsics.fx = msg->k[0];
            intrinsics.fy = msg->k[4];
            intrinsics.coeffs[0] = msg->d[0];
            intrinsics.coeffs[1] = msg->d[1];
            intrinsics.coeffs[2] = msg->d[2];
            intrinsics.coeffs[3] = msg->d[3];
            intrinsics.coeffs[4] = msg->d[4];
        }

        // depth data와 timestamp를 deque에 저장하는 함수
        void depthImageCallback(const sensor_msgs::msg::Image::ConstPtr msg) {
            cv_ptr = cv_bridge::toCvCopy(msg);
            depth_image_ = cv_ptr->image;
            depth_image_.convertTo(depth_image_, CV_32F);

            const int depth_image_deque_size = 15;
            if (depth_image_deque.size() >= depth_image_deque_size){
                depth_image_deque.pop_front();
                depth_image_time_stamp_deque.pop_front();
            }
            depth_image_deque.push_back(depth_image_);
            depth_image_time_stamp_deque.push_back((msg->header.stamp.sec % 1000000) * 1000 + msg->header.stamp.nanosec / 1000000);
            // printf("time gap : %d\n", this->now().nanoseconds() - msg->header.stamp.sec * 1000000000 - msg->header.stamp.nanosec);
            if (now_time2 != 0){
                // printf("time gap : %f\n", ((this->now().nanoseconds() / 1000000) % 1000000000 - now_time2)/1000.0);
            }
            now_time2 = (this->now().nanoseconds() / 1000000) % 1000000000;
        }

        // segmentation 데이터가 발행될 때마다 사람의 위치, 속도, 가속도, 다음 위치를 예측하는 함수
        void segCallback(const std_msgs::msg::Int32MultiArray::SharedPtr msg) {
            // int now_time3 = (this->now().nanoseconds() / 1000000) % 1000000000;
            // calc_processtime(now_time3);
            if (depth_image_.rows <= 0) {
                return;
            }
            std::vector<int> tmp = msg->data;
            const int sync_time = tmp[0];
            const int len = 35; // 1(id) + 34(keypoints.xy)
            const int count = (tmp.size() - 1) / len;

            // keypoints data have to synchronize with depth image
            while(!depth_image_time_stamp_deque.empty() && sync_time - depth_image_time_stamp_deque.front() > 0){
                // printf("Erased!\n");
                depth_image_deque.pop_front();
                depth_image_time_stamp_deque.pop_front();
            }

            // fail when depth_image_time_stamp_deque is empty
            if (depth_image_time_stamp_deque.empty()) {
                return;
            }
            // printf("seg - depth: %d\n", sync_time - depth_image_time_stamp_deque.front());
            // cout << "rows: " << depth_image_deque.front().rows << ", cols: " << depth_image_deque.front().cols << "\n";
            cv::Mat depth_image_tmp = depth_image_deque.front();
            depth_image_deque.pop_front();
            depth_image_time_stamp_deque.pop_front();

            // 각 id마다 새로운 xyzt 정보 추가
            for (size_t i = 0; i < count; i++) {
                // id 값 구하기
                int id = tmp[ 1 + len * i ];
                // id에 해당하는 저장공간 확보
                while (id_keypoint_xyzt_vector.size() < id){
                    id_keypoint_xyzt_vector.emplace_back(17);
                    id_keypoint_velocity_vector.emplace_back(17);
                }
                // printf("total: %d\neach: %d\n", id_keypoint_xyzt_vector.size(), id_keypoint_xyzt_vector[id-1].size());

                // 각 keypoint마다 xyzt 데이터 추가
                for (int j = 0; j < 17; j++) {
                    // OLD_DATA_REFERENCE_TIME ms 이상 차이나는 데이터 삭제하기
                    while (!id_keypoint_xyzt_vector[id-1][j].empty() && sync_time - id_keypoint_xyzt_vector[id-1][j].front().time > OLD_DATA_REFERENCE_TIME) {
                        // printf("Ereased: %dms, %d left\n", sync_time - id_keypoint_xyzt_vector[id-1][j].front().time, id_keypoint_xyzt_vector[id-1][j].size());
                        id_keypoint_xyzt_vector[id-1][j].pop_front();
                    }
                    while (!id_keypoint_velocity_vector[id-1][j].empty() && sync_time - id_keypoint_velocity_vector[id-1][j].front().time > OLD_DATA_REFERENCE_TIME) {
                        id_keypoint_velocity_vector[id-1][j].pop_front();
                    }
                    // pixel 좌표 가져오기
                    int x = tmp[ 2 + len * i + j * 2], y = tmp[ 3 + len * i + j * 2];

                    // keypoint가 있는지 체크
                    if ( x < 0 || y < 0 ) {
                        continue;
                    }
                    // depth가 있는지 체크
                    int depth = depth_image_tmp.at<float>(y, x); // 단위: mm
                    if (depth < 1) {
                        continue;
                    }

                    // 새로운 xyzt 데이터
                    Point new_point = rs2_deproject_pixel_to_point_modified(&intrinsics, x, y, depth, sync_time);
                    // Velocity 계산
                    if (id_keypoint_xyzt_vector[id-1][j].size() > 0) {
                        id_keypoint_velocity_vector[id-1][j].push_back(calculate_velocity(id_keypoint_xyzt_vector[id-1][j].back(), new_point));
                        // // 확인을 위해 최근 Velocity 출력
                        // std::cout << "Velocity: (" 
                        //         << id_keypoint_velocity_vector[id-1][j].back().vx << ", " 
                        //         << id_keypoint_velocity_vector[id-1][j].back().vy << ", " 
                        //         << id_keypoint_velocity_vector[id-1][j].back().vz << ", " 
                        //         << id_keypoint_velocity_vector[id-1][j].back().time << ")\n";
                    }
                    // 새로운 xyzt 데이터 추가 <todo: x, y deprojection 이후의 단위 알아보기>
                    id_keypoint_xyzt_vector[id-1][j].push_back(new_point);
                    // // 확인을 위해 최근 Point 출력
                    // std::cout << "Point: (" 
                    //         << id_keypoint_xyzt_vector[id-1][j].back().x << ", " 
                    //         << id_keypoint_xyzt_vector[id-1][j].back().y << ", " 
                    //         << id_keypoint_xyzt_vector[id-1][j].back().z << ", " 
                    //         << id_keypoint_xyzt_vector[id-1][j].back().time << ")\n";
                }
                // 평균 위치 계산
                Point mean_xyz = calculate_average_point(id_keypoint_xyzt_vector[id-1]);
                // 평균 속도 계산
                Velocity mean_velocity = calculate_average_velocity(id_keypoint_velocity_vector[id-1]);
                // 충돌 시간 및 거리 계산
                std::vector<float> time_distance = find_closest_time_distance(mean_xyz, mean_velocity);
                // Risk score 계산
                if ( 0 < time_distance[0] && time_distance[0] < COLLISION_REFERENCE_TIME && time_distance[1] < COLLISION_REFERENCE_DISTANCE ) {
                    RiskScore risk;
                    risk.id = id;
                    risk.collision_time = time_distance[0];
                    risk.distance = time_distance[1];
                    risk.velocity = std::sqrt(mean_velocity.vx*mean_velocity.vx + mean_velocity.vy*mean_velocity.vy + mean_velocity.vz*mean_velocity.vz);
                    risk.risk_score = risk.velocity / risk.distance * 1000000 / risk.collision_time;
                    risk_list.push_back(risk);
                    printf("id: %d, risk_score: %f\n", risk.id, risk.risk_score);
                    printf("collision time: %f, distance: %f, velocity: %f\n", risk.collision_time, risk.distance, risk.velocity);
                }
                return;
                
                int millisec = tmp[len * (i - 1)];
                // printf("%d\n", millisec);
                std::vector<int> human(tmp.begin() + len * (i - 1) + 1, tmp.begin() + len * i);
                cv::Mat mat_human(human);
                mat_human.convertTo(mat_human, CV_8U);
                mat_human = mat_human.reshape(1, depth_image_.rows);
                // removeOutline(mat_human);
                // printf("time gap : %f\n", ((this->now().nanoseconds() / 1000000) % 1000000000 - millisec) / 1000.0);
                cv::Mat result;
                if (depth_image_deque.size() >= 3){
                    result = human_filter(depth_image_deque[depth_image_deque.size() - 1], mat_human);
                }
                else{
                    printf("check!\n");
                    result = human_filter(depth_image_, mat_human);
                }
                
                // statistical outlier removal
                removeOutliers(result);

                point_cloud_ptr = std::make_shared<pcl::PointCloud<pcl::PointXYZRGB>>();
                for (int y = 0; y < result.rows; y++) {
                    for (int x = 0; x < result.cols; x++) {
                        float depth = result.at<float>(y, x) / 1000; // meter
                        if (depth > 0) {
                            cv::Vec3b pixel = color_image_.at<cv::Vec3b>(y, x);
                            int blue = pixel[0];
                            int green = pixel[1];
                            int red = pixel[2];
                            pcl::PointXYZRGB point;
                            rs2_deproject_pixel_to_point(point, &intrinsics, x, y, depth);
                            // point.r = red;
                            // point.g = green;
                            // point.b = blue;
                            point_cloud_ptr->points.push_back(point);
                        }
                    }
                }

                // // Create the filtering object: downsample the dataset using a leaf size of 1cm
                // pcl::VoxelGrid<pcl::PointXYZRGB> vg;
                // vg.setInputCloud(point_cloud_ptr);
                // vg.setLeafSize(0.01f, 0.01f, 0.01f);
                // vg.filter(*point_cloud_ptr);
            }

            const int xyzt_deque_size = 4;
            if (xyzt_deque.size() >= xyzt_deque_size){
                xyzt_deque.pop_front();
            }
            xyzt_deque.push_back(calculateAverage(*point_cloud_ptr, depth_image_time_stamp_deque[depth_image_time_stamp_deque.size() - 3]));

            // AverageVelocity
            if (xyzt_deque.size() == xyzt_deque_size){
                // 속도 및 가속도 계산
                std::vector<Velocity> velocities;
                Velocity meanVelocity;
                Acceleration meanAcceleration;

                for (int i = 0; i < xyzt_deque_size - 1; ++i) {
                    Velocity v = calculateVelocity(xyzt_deque[i], xyzt_deque[i + 1]);
                    velocities.push_back(v);
                    meanVelocity.vx += v.vx;
                    meanVelocity.vy += v.vy;
                    meanVelocity.vz += v.vz;
                }
                meanVelocity.vx /= xyzt_deque_size - 1;
                meanVelocity.vy /= xyzt_deque_size - 1;
                meanVelocity.vz /= xyzt_deque_size - 1;

                for (int i = 0; i < xyzt_deque_size - 2; ++i) {
                    Acceleration a = calculateAcceleration(velocities[i], velocities[i + 1], xyzt_deque[i + 1].time - xyzt_deque[i].time);
                    meanAcceleration.ax += a.ax;
                    meanAcceleration.ay += a.ay;
                    meanAcceleration.az += a.az;
                }
                meanAcceleration.ax /= xyzt_deque_size - 2;
                meanAcceleration.ay /= xyzt_deque_size - 2;
                meanAcceleration.az /= xyzt_deque_size - 2;
                
                // i => 0.1sec
                float min_dist = 999999999.0;
                int min_idx;
                for (int i = 1; i <= 11; i++) {
                    Point after_i_seconds;
                    after_i_seconds.x = xyzt_deque.back().x + (meanVelocity.vx * i * 0.1) + 0.5 * (meanAcceleration.ax * i * i * 0.01);
                    after_i_seconds.y = xyzt_deque.back().y + (meanVelocity.vy * i * 0.1) + 0.5 * (meanAcceleration.ay * i * i * 0.01);
                    after_i_seconds.z = xyzt_deque.back().z + (meanVelocity.vz * i * 0.1) + 0.5 * (meanAcceleration.az * i * i * 0.01);
                    float dist = std::sqrt(after_i_seconds.x * after_i_seconds.x + after_i_seconds.y * after_i_seconds.y + after_i_seconds.z * after_i_seconds.z);
                    // vel = std::sqrt(meanVelocity.vx * meanVelocity.vx + meanVelocity.vy * meanVelocity.vy + meanVelocity.vz * meanVelocity.vz);
                    // acc = std::sqrt(meanAcceleration.ax * meanAcceleration.ax + meanAcceleration.ay * meanAcceleration.ay + meanAcceleration.az * meanAcceleration.az);
                    if (dist < min_dist && meanVelocity.vx < -0.3 && meanAcceleration.ax < 0.0 && meanAcceleration.ax > -10.0) {
                        min_dist = dist;
                        min_idx = i;
                    }
                }
                if (min_dist < 1.0) {
                    // printf("kinemetics : %fm, %fm/s, %fm/s^2\n", min_dist, meanVelocity.vx, meanAcceleration.ax);
                    printf("danger : %f\n", -1 * meanVelocity.vx / min_dist / min_idx);
                }

                // 결과 출력
                // std::cout << "\nResults:\n";
                // printf("time gap : %f\n", ((this->now().nanoseconds() / 1000000) % 1000000000 - xyzt_deque[2].time) / 1000.0);
                // printf("time gap2 : %f\n", (xyzt_deque[2].time - xyzt_deque[1].time) / 1000.0);
                // printf("vel : %f, %f, %f\n", meanVelocity.vx, meanVelocity.vy, meanVelocity.vz);
                // printf("acc : %f, %f, %f\n", meanAcceleration.ax, meanAcceleration.ay, meanAcceleration.az);
            }

            // // flight, cycle time
            // if (now_time != 0) {
            //     // flight_time.push_back(((this->now().nanoseconds() / 1000000) % 1000000000 - xyzt_deque.back().time) / 1000.0);
            //     flight_time.push_back(((this->now().nanoseconds() / 1000000) % 1000000000 - xyzt_deque.front().time) / 1000.0);
            //     // cycle_time.push_back(((this->now().nanoseconds() / 1000000) % 1000000000 - now_time) / 1000.0);
            //     cycle_time.push_back((xyzt_deque.back().time - xyzt_deque[xyzt_deque.size() - 2].time) / 1000.0);
            //     // printf("%d, %f, %f\n", flight_time.size(), calculateAverage(flight_time), calculateAverage(cycle_time));
            //     if (flight_time.size() == 1000){
            //         const std::string filename = "flight_time.txt";
            //         std::ofstream outFile(filename);
            //         if (!outFile.is_open()) {
            //             std::cerr << "파일을 열 수 없습니다." << std::endl;
            //             return;
            //         }
            //         for (const auto& value : flight_time) {
            //             outFile << value << std::endl;
            //         }
            //         outFile.close();
            //         printf("%d, %f, %f\n", flight_time.size(), calculateAverage(flight_time), calculateAverage(cycle_time));
            //     }
            // }
            // now_time = (this->now().nanoseconds() / 1000000) % 1000000000;

            // // pcl visualizer
            // point_cloud_ptr->width = point_cloud_ptr->size();
            // point_cloud_ptr->height = 1;
            // point_cloud_ptr->is_dense = true;
            // viewer->updatePointCloud(point_cloud_ptr, "sample cloud");
            // viewer->spinOnce(100);
            // std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // // processing time
            // processTime.push_back(((this->now().nanoseconds() / 1000000) % 1000000000 - now_time3) / 1000.0);
            // if (processTime.size() == 1000){
            //     const std::string filename = "processTime.txt";
            //     std::ofstream outFile(filename);
            //     if (!outFile.is_open()) {
            //         std::cerr << "파일을 열 수 없습니다." << std::endl;
            //         return;
            //     }
            //     for (const auto& value : processTime) {
            //         outFile << value << std::endl;
            //     }
            //     outFile.close();
            //     printf("%f\n", ((this->now().nanoseconds() / 1000000) % 1000000000 - now_time3) / 1000.0);
            // }
        }

        // void xyzt_publish() {
        //     // Create a message object
        //     auto message = std_msgs::msg::Float32MultiArray();

        //     // Create a deque of vectors
        //     std::deque<std::vector<float>> dequeData = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}, {7.0, 8.0, 9.0}};

        //     // Flatten the deque into a single vector
        //     std::vector<float> flattenedData;
        //     for (const auto &vec : dequeData)
        //     {
        //     flattenedData.insert(flattenedData.end(), vec.begin(), vec.end());
        //     }

        //     // Set the data field of the message
        //     message.data = flattenedData;

        //     // Publish the message
        //     xyzt_publisher_->publish(message);
        // }
};

// 노드 활성화
int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DistNode>());
    rclcpp::shutdown();
    return 0;
}
