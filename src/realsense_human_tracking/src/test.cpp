#include <opencv2/opencv.hpp>

void removeOutliers(cv::Mat& mat, double threshold) {
    // 평균과 표준 편차 계산
    cv::Scalar mean, stddev;
    cv::meanStdDev(mat, mean, stddev);

    // Z-score를 계산하고, 임계값 이상인 값을 0으로 설정
    mat.forEach<int>([&](int& pixel, const int* position) -> void {
        double zScore = (pixel - mean[0]) / stddev[0];
        if (std::abs(zScore) > threshold) {
            pixel = 0;
        }
    });
}

int main() {
    // mat_human을 적절히 초기화
    cv::Mat mat_human = cv::Mat::ones(100, 100, CV_32S);  // 예시로 100x100 크기의 정규분포에서 생성된 매트릭스

    // 처리 전 출력
    std::cout << "Before Processing:\n" << mat_human << "\n";

    // Outlier 제거
    double threshold = 2.0;  // Z-score의 절대값이 이 값보다 크면 Outlier로 판단
    removeOutliers(mat_human, threshold);

    // 처리 후 출력
    std::cout << "\nAfter Processing:\n" << mat_human << "\n";

    return 0;
}
