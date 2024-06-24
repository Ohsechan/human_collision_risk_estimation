# Human Collision Risk Estimation
## 실행 방법

<pre><code># Turn on the realsense camera
ros2 launch realsense2_camera rs_launch.py camera_namespace:=AMR camera_name:=D435i depth_module.profile:=640x480x30 rgb_camera.profile:=640x480x30 enable_sync:=true align_depth.enable:=true
# Human pose estimation LSTM + Seg
ros2 run image_processing image_processing_node
# Danger index calculation
ros2 run risk_estimation risk_estimation_node
</code></pre>
