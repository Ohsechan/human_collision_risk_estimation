## 1. 프로젝트 설명
### 1-1. 수상 이력
- WCRC 2023 Data Contest 1위 과학기술정보통신부장관상
### 1-2. 개발환경
- Ubuntu 22.04 Desktop
- ROS2 Humble

## 2. 데모 영상
[![사람의 point cloud data 시각화](http://img.youtube.com/vi/v3hmKNEFw_o/0.jpg)](https://www.youtube.com/watch?v=v3hmKNEFw_o&list=PLx5EbqT-6Y08K1ZaK8a7qJ8qOc2PsTDvh)

## 3. Quick start

### 3-1. Turn on the realsense camera
<pre><code>ros2 launch realsense2_camera rs_launch.py depth_module.profile:=640x480x30 rgb_camera.profile:=640x480x30</code></pre>

### 3-2. Human pose estimation LSTM + Seg
<pre><code>ros2 launch depth_example merge_lstm_seg.launch.py</code></pre>

### 3-3. Danger index calculation
<pre><code>ros2 run realsense_human_tracking human_pcl</code></pre>

## 4. 문서
https://docs.google.com/document/d/1YqBIqOUxnI7RKZ2SeHtM_VViJVNKV7SAQmxj7HR6lik/edit#heading=h.2gazcsgmxkub
