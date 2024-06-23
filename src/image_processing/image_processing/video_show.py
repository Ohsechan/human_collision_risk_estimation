import cv2
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_prefix

def find_package_path(package_name):
    package_path = get_package_prefix(package_name)
    package_path = os.path.dirname(package_path)
    package_path = os.path.dirname(package_path)
    package_path = os.path.join(package_path, "src", package_name)
    return package_path

package_path = find_package_path('image_processing')
yolo_model = YOLO(os.path.join(package_path,'models','yolov8n-pose.pt'))

# avi 파일 경로
video_path = '\
/home/ohbuntu22/human_collision_risk_estimation/src/image_processing/dataset_action_split/test/Sitting/video_4_flip.avi\
'
# VideoCapture 객체 생성
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = yolo_model.predict(frame,
                    verbose=False, # no print
                    max_det=1,
                    device="cuda:0")
        
        # 원본 프레임과 YOLO 결과를 가로로 병합
        combined_frame = cv2.hconcat([frame, results[0].plot()])

        # 병합된 프레임을 하나의 윈도우에 표시
        cv2.imshow("Original and YOLO Detection", combined_frame)
        # 'q' 키를 누르면 종료
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()