import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np

# 1. 라이브러리 및 모델 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 2. 파일 경로 설정
# 촬영한 영상들이 있는 폴더 경로
VIDEO_DIR = r'C:\Users\1324h\OneDrive\Desktop\hip_rom_project\videos'
# 추출된 CSV 파일을 저장할 폴더 경로
CSV_OUTPUT_DIR = r'C:\Users\1324h\Desktop\hip_rom_project\csv_data'

#출력 폴더가 없으면 생성
if not os.path.exists(CSV_OUTPUT_DIR):
    os.makedirs(CSV_OUTPUT_DIR)

# 3. MediaPipe Pose 모델 설정
# model_complexity=1: 모델 복잡도 (0, 1, 2 중 1이 균형이 가장 좋음)
# min_detection_confidence: 사람 감지 최소 신뢰도
# min_tracking_confidence: 키포인트 추적 최소 신뢰도
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    # VIDEO_DIR 폴더에 있는 모든 mp4 파일 목록 가져오기
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]

    # 파일 목록을 순차적으로 처리
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        
        # 파일명에서 확장자를 제거하여 CSV 파일명 생성
        csv_file_name = os.path.splitext(video_file)[0] + '.csv'
        csv_path = os.path.join(CSV_OUTPUT_DIR, csv_file_name)

        print(f"Processing video: {video_file}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_file}.")
            continue

        all_landmarks_data = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # BGR 이미지를 RGB로 변환 (MediaPipe는 RGB 선호)
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Pose 추정 수행
            results = pose.process(frame)

            # 다시 그리기 가능하도록 설정하고 RGB를 BGR로 변환
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 랜드마크 추출 및 저장
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                frame_data = {}

                # 모든 랜드마크의 x, y, z, visibility를 저장
                # 특히 z좌표는 3D 각도 계산에 필수적임
                for i, lm in enumerate(landmarks):
                    frame_data[f'lm_{i}_x'] = lm.x
                    frame_data[f'lm_{i}_y'] = lm.y
                    frame_data[f'lm_{i}_z'] = lm.z
                    frame_data[f'lm_{i}_visibility'] = lm.visibility
                
                # 추가 정보 (예: 현재 프레임 번호)
                frame_data['frame'] = cap.get(cv2.CAP_PROP_POS_FRAMES)
                all_landmarks_data.append(frame_data)
                
                # 추출된 키포인트를 화면에 시각화 (확인용)
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # 결과 화면 보여주기 (속도 저하가 싫다면 이 부분을 주석 처리)
            cv2.imshow('MediaPipe Pose Landmark Extraction', frame)

            # 'q' 키를 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # 추출된 데이터를 Pandas DataFrame으로 변환하고 CSV 파일로 저장
        if all_landmarks_data:
            df_landmarks = pd.DataFrame(all_landmarks_data)
            df_landmarks.to_csv(csv_path, index=False)
            print(f"Landmarks saved to {csv_path}")
        else:
            print(f"No landmarks found for video {video_file}.")

print("All videos have been processed.")
