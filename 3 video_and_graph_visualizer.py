import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# 1. 파일 경로 설정
VIDEO_DIR = r'C:\Users\1324h\OneDrive\Desktop\hip_rom_project\videos'
PROCESSED_DATA_DIR = r'C:\Users\1324h\OneDrive\Desktop\hip_rom_project\processed_data'
VISUALIZATION_OUTPUT_DIR = 'visualization_videos'

# 입력 비디오 파일, CSV 파일 이름 설정
video_filename = '20250804_101457_1_2 hip abd+ext.rot PROM.mp4'
csv_filename = '20250804_101457_1_2 hip abd+ext.rot PROM_processed.csv'

# 전체 경로 생성
video_path = os.path.join(VIDEO_DIR, video_filename)
csv_path = os.path.join(PROCESSED_DATA_DIR, csv_filename)
output_path = os.path.join(VISUALIZATION_OUTPUT_DIR, f'visualized_{video_filename}')

# 출력 폴더가 없으면 생성
if not os.path.exists(VISUALIZATION_OUTPUT_DIR):
    os.makedirs(VISUALIZATION_OUTPUT_DIR)

# 2. 비디오 및 데이터 읽기
cap = cv2.VideoCapture(video_path)
df = pd.read_csv(csv_path)

if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

# 3. 비디오 속성 가져오기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 4. 비디오 저장을 위한 VideoWriter 객체 생성 (오른쪽에 그래프를 붙이므로 너비 2배)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

# 5. matplotlib 그래프 설정 (영상 루프 밖에서 미리 설정)
plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.tight_layout(pad=3.0)

# 6. 비디오 프레임별 시각화 루프
frame_number = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 7. 현재 프레임에 대한 그래프 생성
    ax1.clear()
    ax2.clear()
    ax3.clear()

    # --- 1번 그래프: 굴곡 각도 ---
    ax1.set_title('Hip Flexion Angle Over Time')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_ylim(0, 180)
    ax1.plot(df.index, df['left_hip_flex_ext_angle'], label='Left Flexion', color='blue')
    ax1.plot(df.index, df['right_hip_flex_ext_angle'], label='Right Flexion', color='skyblue')
    ax1.legend(loc='upper right')
    ax1.grid(True)

    # --- 2번 그래프: 외전 각도 ---
    ax2.set_title('Hip Abduction Angle Over Time')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_ylim(0, 180)
    ax2.plot(df.index, df['left_hip_abd_add_angle'], label='Left Abduction', color='red')
    ax2.plot(df.index, df['right_hip_abd_add_angle'], label='Right Abduction', color='orange')
    ax2.legend(loc='upper right')
    ax2.grid(True)

    # --- 3번 그래프: (발레 특화) 90도 굴곡 상태에서 외전&외회전 복합움직임 각도 ---
    ax3.set_title('Hip External Rotation Angle Over Time')
    ax3.set_ylabel('Angle (degrees)')
    ax3.set_xlabel('Frame Number')
    ax3.set_ylim(-180, 180)
    ax3.plot(df.index, df['left_hip_external_rotation_approx_angle'], label='Left External Rotation', color='green')
    ax3.plot(df.index, df['right_hip_external_rotation_approx_angle'], label='Right External Rotation', color='lightgreen')
    ax3.legend(loc='upper right')
    ax3.grid(True)

    # 현재 프레임을 강조하는 점 또는 수직선 그리기
    if frame_number < len(df):
        # 각 서브플롯에 수직선 그리기
        ax1.axvline(x=frame_number, color='red', linestyle='--', linewidth=1)
        ax2.axvline(x=frame_number, color='red', linestyle='--', linewidth=1)
        ax3.axvline(x=frame_number, color='red', linestyle='--', linewidth=1)

        # --- 실시간 수치 텍스트 표시 ---
        # 굴곡 각도
        flex_left = df.loc[frame_number, 'left_hip_flex_ext_angle']
        flex_right = df.loc[frame_number, 'right_hip_flex_ext_angle']
        ax1.text(0.02, 0.05, f'Flex L:{flex_left:.2f}°\nR:{flex_right:.2f}°',
                 transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.7))

        # 외전 각도
        abd_left = df.loc[frame_number, 'left_hip_abd_add_angle']
        abd_right = df.loc[frame_number, 'right_hip_abd_add_angle']
        ax2.text(0.02, 0.05, f'Abd L:{abd_left:.2f}°\nR:{abd_right:.2f}°',
                 transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.7))

        # 외회전 각도
        rot_left = df.loc[frame_number, 'left_hip_external_rotation_approx_angle']
        rot_right = df.loc[frame_number, 'right_hip_external_rotation_approx_angle']
        ax3.text(0.02, 0.05, f'Rot L:{rot_left:.2f}°\nR:{rot_right:.2f}°',
                 transform=ax3.transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    # 8. matplotlib 그래프를 이미지로 변환
    fig.canvas.draw()
    graph_image_argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    graph_image_reshaped = graph_image_argb.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    graph_image_bgr = cv2.cvtColor(graph_image_reshaped, cv2.COLOR_BGRA2BGR)
    
    # 9. 그래프 이미지 크기 조정 (비디오 높이와 일치시키기)
    graph_image_bgr = cv2.resize(graph_image_bgr, (width, height))
    
    # 10. 비디오 프레임과 그래프 이미지 합치기
    combined_frame = np.hstack((frame, graph_image_bgr))

    # 11. 최종 프레임을 비디오 파일에 쓰기
    out.write(combined_frame)

    frame_number += 1

    cv2.imshow('Video with Graph', combined_frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 12. 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()
plt.close(fig)

print(f"Visualization video saved to {output_path}")
