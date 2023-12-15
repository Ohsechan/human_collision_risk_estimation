# Human Collision Risk Estimation
## 1. 프로젝트 요약
### 1-1. 설명
자율 이동 로봇(AMR, Autonomous Mobile Robots)의 발전에 따라 사람과 로봇이 같은 공간을 사용하는 빈도가 증가하게 될 것이다. 이때, 자율 이동 로봇은 사람과의 충돌에 대한 안전을 최우선으로 고려하여 주행해야한다. 본 프로젝트는 사람과의 충돌 위험성을 계산하고 돌방상황을 예측하여 충돌 회피에 도움을 줄 수 있는 인지 시스템을 제안한다. Realsense depth camera를 활용하여 앉아있는 사람과 서있는 사람을 구분하고, 서있는 사람에 대한 3D point cloud 정보를 바탕으로 자율 이동 로봇과 충돌 위험이 있는지, 그리고 그 위험이 얼마나 높은지를 수치화하여 제공한다.
### 1-2. 수상 이력
- WCRC 2023 Data Contest 1위 과학기술정보통신부장관상
### 1-3. 개발 규모
- 인원 : 4명
- 일시 : 2023.11.01 ~ 2023.11.20
### 1-4. 개발환경
- Ubuntu 22.04 Desktop
- ROS2 Humble
- C++ : OpenCV, PCL
- Python : Yolov8, PyTorch
### 1-5. 하드웨어
- Realsense D435i

## 2. 프로젝트 구현
### 2-1. 데모 영상
[![youtube playlist](http://img.youtube.com/vi/GwNal6DAgPs/0.jpg)](https://www.youtube.com/playlist?list=PLx5EbqT-6Y08K1ZaK8a7qJ8qOc2PsTDvh)
### 2-2. 시스템 구성도
![wcrc system diagram drawio](https://github.com/Ohsechan/human_collision_risk_estimation/assets/77317210/32f93bea-47a4-48eb-9c44-dbe96715c17e)
### 2-3. 순서도
![wcrc sequence diagram drawio](https://github.com/Ohsechan/human_collision_risk_estimation/assets/77317210/6c6a911b-6b02-4e31-898d-8687dbfaaad2)
### 2-4. 노드 기준 연산시간
![wcrc node diagram drawio](https://github.com/Ohsechan/human_collision_risk_estimation/assets/77317210/8129fad9-134b-4b58-b76d-dce48ac6a74a)
### 2-5. 위치 및 시간 자료구조
![wcrc data store diagram drawio](https://github.com/Ohsechan/human_collision_risk_estimation/assets/77317210/8bc5d9e8-512d-426c-96a5-178b2d8267c5)
### 2-6. 실시간 위치 예측
![wcrc timeline diagram drawio](https://github.com/Ohsechan/human_collision_risk_estimation/assets/77317210/a3fde8ac-6f08-4347-90e3-7b9359ed1cf2)


## 3. 실행 방법

<pre><code># Turn on the realsense camera
ros2 launch realsense2_camera rs_launch.py depth_module.profile:=640x480x30 rgb_camera.profile:=640x480x30
# Human pose estimation LSTM + Seg
ros2 launch depth_example merge_lstm_seg.launch.py
# Danger index calculation
ros2 run realsense_human_tracking human_pcl
</code></pre>

## 4. 문서
https://docs.google.com/document/d/1YqBIqOUxnI7RKZ2SeHtM_VViJVNKV7SAQmxj7HR6lik/edit#heading=h.2gazcsgmxkub
