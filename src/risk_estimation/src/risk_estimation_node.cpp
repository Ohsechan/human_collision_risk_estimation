
#include "rclcpp/rclcpp.hpp"
#include "hcre_msgs/msg/risk_score.hpp"
#include "hcre_msgs/msg/risk_score_array.hpp"
#include "hcre_msgs/msg/pose_tracking.hpp"
#include "hcre_msgs/msg/pose_tracking_array.hpp"

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

// Parameters
const int OLD_DATA_REFERENCE_TIME       = 500;      // Unit: ms
const int COLLISION_REFERENCE_TIME      = 10000;    // Unit: ms
const int COLLISION_REFERENCE_DISTANCE  = 1000;     // Unit: mm

// Realsesne depth camera의 내부 파라미터 구조
struct rs2_intrinsics {
    float         ppx;       /**< Horizontal coordinate of the principal point of the image, as a pixel offset from the left edge */
    float         ppy;       /**< Vertical coordinate of the principal point of the image, as a pixel offset from the top edge */
    float         fx;        /**< Focal length of the image plane, as a multiple of pixel width */
    float         fy;        /**< Focal length of the image plane, as a multiple of pixel height */
    float         coeffs[5]; /**< Distortion coefficients */
};

// Position & timestamp
struct Point {
    float x=0, y=0, z=0;
    int time=0;
};

// Velocity & timestamp
struct Velocity {
    float vx=0, vy=0, vz=0;
    int time=0;
};

// Node 정의
class RiskEstimationNode : public rclcpp::Node {
    public:
        // publisher 및 subscriber 정의
        RiskEstimationNode() : Node("risk_estimation_node") {
            camera_info_subscriber_ = create_subscription<sensor_msgs::msg::CameraInfo>(
                "/AMR/D435i/aligned_depth_to_color/camera_info",
                10,
                std::bind(&RiskEstimationNode::cameraInfoCallback, this, std::placeholders::_1)
            );
            depth_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/AMR/D435i/aligned_depth_to_color/image_raw",
                10,
                std::bind(&RiskEstimationNode::depthImageCallback, this, std::placeholders::_1)
            );
            pose_tracking_data_subscription_ = this->create_subscription<hcre_msgs::msg::PoseTrackingArray>(
                "/image_processing/pose_tracking",
                10,
                std::bind(&RiskEstimationNode::segCallback, this, std::placeholders::_1)
            );
            risk_score_array_publisher_ = this->create_publisher<hcre_msgs::msg::RiskScoreArray>(
                "/risk_estimation/risk_score_array",
                10);
        }

    private:
        // 클래스에서 사용할 변수 정의
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_subscriber_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_subscription_;
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr color_subscription_;
        rclcpp::Subscription<hcre_msgs::msg::PoseTrackingArray>::SharedPtr pose_tracking_data_subscription_;
        rclcpp::Publisher<hcre_msgs::msg::RiskScoreArray>::SharedPtr risk_score_array_publisher_;
        cv_bridge::CvImagePtr cv_ptr;
        cv::Mat depth_image_;
        std::deque<cv::Mat> depth_image_deque;
        std::deque<int> depth_image_time_stamp_deque;
        std::vector<std::vector<std::deque<Point>>> id_keypoint_xyzt_vector;
        std::vector<std::vector<std::deque<Velocity>>> id_keypoint_velocity_vector;
        rs2_intrinsics intrinsics;
        int now_time = 0;
        std::vector<float> flight_time, cycle_time;
        int now_time2 = 0;
        std::vector<unsigned long long> processTime; // for calculate process time
        std::vector<int> total_processTime; // for calculate process time

        void calc_processtime(unsigned long long now_time_input) {
            processTime.push_back(this->now().nanoseconds() % 1000000000000 - now_time_input); // 12자리수, 단위: ns
            if (processTime.size() == 1050){
                const std::string filename = "processTime.txt";
                std::ofstream outFile(filename);
                if (!outFile.is_open()) {
                    std::cerr << "파일을 열 수 없습니다." << std::endl;
                    return;
                }
                for (auto it = processTime.begin() + 50; it != processTime.end(); ++it) {
                    outFile << *it << std::endl;
                }
                outFile.close();
            }
        }

        void calc_total_processtime(int sync_time) {
            total_processTime.push_back((this->now().nanoseconds()/1000000)%1000000000 - sync_time); // 12자리수, 단위: ns
            if (total_processTime.size() == 1050){
                const std::string filename = "total_processTime.txt";
                std::ofstream outFile(filename);
                if (!outFile.is_open()) {
                    std::cerr << "파일을 열 수 없습니다." << std::endl;
                    return;
                }
                for (auto it = total_processTime.begin() + 50; it != total_processTime.end(); ++it) {
                    outFile << *it << std::endl;
                }
                outFile.close();
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

        // camera의 intrinsic parameter를 받아오는 함수
        void cameraInfoCallback(const std::shared_ptr<const sensor_msgs::msg::CameraInfo> msg) {
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
        void depthImageCallback(const std::shared_ptr<const sensor_msgs::msg::Image> msg) {
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
        void segCallback(const std::shared_ptr<const hcre_msgs::msg::PoseTrackingArray> msg) {
            // unsigned long long now_time3 = this->now().nanoseconds() % 1000000000000; // 12자리, 단위: ns <todo: 시간 계산용>
            if (depth_image_.rows <= 0) {
                return;
            }
            const int sync_time = msg->timestamp;

            // keypoints data have to synchronize with depth image
            while(!depth_image_time_stamp_deque.empty() && sync_time - depth_image_time_stamp_deque.front() > 0){
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

            // 각 사람의 risk_score를 저장하여 publish할 msg
            auto risk_score_array_msg = std::make_shared<hcre_msgs::msg::RiskScoreArray>();

            // calc_processtime(now_time3); // <todo: 시간 계산용>

            // 각 id마다 새로운 xyzt 정보 추가
            for (size_t i = 0; i < msg->posetracking.size(); i++) {
                // id 값 구하기
                size_t id = msg->posetracking[i].id;
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
                        id_keypoint_xyzt_vector[id-1][j].pop_front();
                    }
                    while (!id_keypoint_velocity_vector[id-1][j].empty() && sync_time - id_keypoint_velocity_vector[id-1][j].front().time > OLD_DATA_REFERENCE_TIME) {
                        id_keypoint_velocity_vector[id-1][j].pop_front();
                    }

                    // pixel 좌표 가져오기
                    int x = msg->posetracking[i].x[j], y = msg->posetracking[i].y[j];

                    // keypoint가 있는지 체크
                    if ( x <= 0 || y <= 0 ) {
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
                    // 새로운 xyzt 데이터 추가
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
                hcre_msgs::msg::RiskScore risk;
                risk.id = id;
                risk.pose = msg->posetracking[i].pose;
                risk.velocity = std::sqrt(mean_velocity.vx*mean_velocity.vx + mean_velocity.vy*mean_velocity.vy + mean_velocity.vz*mean_velocity.vz);
                if ( 0 < time_distance[0] && time_distance[0] < COLLISION_REFERENCE_TIME && time_distance[1] < COLLISION_REFERENCE_DISTANCE ) {
                    risk.time_to_collision = time_distance[0] / 1000; // second
                    risk.minimum_distance = time_distance[1] / 1000; // meter
                    risk.risk_score = risk.velocity / risk.minimum_distance / risk.time_to_collision;
                    if (risk.pose) {
                        risk.risk_score *= 2;
                    }
                }
                else {
                    risk.time_to_collision = COLLISION_REFERENCE_TIME / 1000; // second
                    risk.minimum_distance = std::sqrt(mean_xyz.x * mean_xyz.x + mean_xyz.y * mean_xyz.y + mean_xyz.z * mean_xyz.z) / 1000; // meter
                    risk.risk_score = 0;
                }
                risk_score_array_msg->scores.push_back(risk);
                // std::cout <<"id: " << risk.id << ", risk_score: " << risk.risk_score << "\n";
                // std::cout <<"collision time: " << risk.time_to_collision << ", distance: " << risk.minimum_distance << ", velocity: " << risk.velocity << "\n";
            }
            risk_score_array_publisher_->publish(*risk_score_array_msg);
            // calc_total_processtime(sync_time);
        }
};

// 노드 활성화
int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RiskEstimationNode>());
    rclcpp::shutdown();
    return 0;
}
