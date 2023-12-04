# Human Collision Risk Estimation
## 1. 프로젝트 설명
### 1-1. 요약
자율 이동 로봇(AMR, Autonomous Mobile Robots)의 발전에 따라 사람과 로봇이 같은 공간을 사용하는 빈도가 증가하게 될 것이다. 이때, 자율 이동 로봇은 사람과의 충돌에 대한 안전을 최우선으로 고려하여 주행해야한다. 본 프로젝트는 사람과의 충돌 위험성을 계산하고 돌방상황을 예측하여 충돌 회피에 도움을 줄 수 있는 인지 시스템을 제안한다. Realsense depth camera를 활용하여 앉아있는 사람과 서있는 사람을 구분하고, 서있는 사람에 대한 3D point cloud 정보를 바탕으로 자율 이동 로봇과 충돌 위험이 있는지, 그리고 그 위험이 얼마나 높은지를 수치화하여 제공한다.
### 1-2. 수상 이력
- WCRC 2023 Data Contest 1위 과학기술정보통신부장관상
### 1-3. 개발환경
- Ubuntu 22.04 Desktop
- ROS2 Humble
- C++ : OpenCV, PCL
- Python : Yolov8, PyTorch
### 1-4. 하드웨어
- Realsense D435i

## 2. 데모 영상
[![사람의 point cloud data 시각화](http://img.youtube.com/vi/v3hmKNEFw_o/0.jpg)](https://www.youtube.com/watch?v=v3hmKNEFw_o&list=PLx5EbqT-6Y08K1ZaK8a7qJ8qOc2PsTDvh)

## 3. 실행 방법

### 3-1. Turn on the realsense camera
<pre><code>ros2 launch realsense2_camera rs_launch.py depth_module.profile:=640x480x30 rgb_camera.profile:=640x480x30</code></pre>

### 3-2. Human pose estimation LSTM + Seg
<pre><code>ros2 launch depth_example merge_lstm_seg.launch.py</code></pre>

### 3-3. Danger index calculation
<pre><code>ros2 run realsense_human_tracking human_pcl</code></pre>

## 4. 문서
https://docs.google.com/document/d/1YqBIqOUxnI7RKZ2SeHtM_VViJVNKV7SAQmxj7HR6lik/edit#heading=h.2gazcsgmxkub
