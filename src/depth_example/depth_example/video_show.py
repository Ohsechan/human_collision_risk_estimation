import cv2

def play_video(video_path):
    # VideoCapture 객체 생성
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # 프레임을 보여줌
            cv2.imshow('Video Playback', frame)
            # 'q' 키를 누르면 종료
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

# avi 파일 경로
video_path = '\
/home/ohbuntu22/human_collision_risk_estimation/src/depth_example/dataset_action_split/train/Fall Down/video_6.avi\
'

# 비디오 재생 함수 호출
play_video(video_path)
