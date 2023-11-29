# human_collision_risk_estimation

### 1. Turn on the realsense camera
<pre><code>ros2 launch realsense2_camera rs_launch.py depth_module.profile:=640x480x30 rgb_camera.profile:=640x480x30</code></pre>

### 2. Human pose estimation LSTM + Seg
<pre><code>ros2 launch depth_example merge_lstm_seg.launch.py</code></pre>

### 3. Danger index calculation
<pre><code>ros2 run realsense_human_tracking human_pcl</code></pre>

### Human Pose Estimation
https://docs.google.com/document/d/1927AtABfKBExAzwcSmCpmuVZfE72IKWgzNLeWiTiSR8/edit
