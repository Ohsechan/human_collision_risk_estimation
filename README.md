# human_collision_risk_estimation

## 1. 프로젝트 설명
### 1-1. 수상 이력
- WCRC 2023 Data Contest 1위 과학기술정보통신부장관상
### 1-2. 개발환경
- Ubuntu 22.04 Desktop
- ROS2 Humble

## 2. Quick start

### 2-1. Turn on the realsense camera
<pre><code>ros2 launch realsense2_camera rs_launch.py depth_module.profile:=640x480x30 rgb_camera.profile:=640x480x30</code></pre>

### 2-2. Human pose estimation LSTM + Seg
<pre><code>ros2 launch depth_example merge_lstm_seg.launch.py</code></pre>

### 2-3. Danger index calculation
<pre><code>ros2 run realsense_human_tracking human_pcl</code></pre>

### 2-4. Human Pose Estimation
https://docs.google.com/document/d/1927AtABfKBExAzwcSmCpmuVZfE72IKWgzNLeWiTiSR8/edit
