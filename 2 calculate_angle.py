import pandas as pd
import numpy as np
import os

# 1. 파일 경로 설정
CSV_INPUT_DIR = r'C:\Users\1324h\Desktop\hip_rom_project\csv_data'
PROCESSED_OUTPUT_DIR = r'processed_data'

# 출력 폴더가 없으면 생성
if not os.path.exists(PROCESSED_OUTPUT_DIR):
    os.makedirs(PROCESSED_OUTPUT_DIR)

# 2. 랜드마크 인덱스 정의
# MediaPipe Pose 공식 문서에 나온 인덱스
# 굴곡/신전 각도 계산에 필요한 랜드마크
LANDMARKS = {
    'LEFT_HIP': 23, 'LEFT_KNEE': 25, 'LEFT_ANKLE': 27,
    'RIGHT_HIP': 24, 'RIGHT_KNEE': 26, 'RIGHT_ANKLE': 28,
    'LEFT_SHOULDER': 11, 'RIGHT_SHOULDER': 12
}

# 3. 각도 계산 함수 정의
def calculate_angle(a, b, c):
    """ 세 점(a, b, c)을 이용하여 b를 꼭짓점으로 하는 각도(도 단위)를 계산하는 함수 """
    # 세 점의 3D 좌표를 NumPy 배열로 변환
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # 벡터 BA와 BC를 계산
    ba = a - b
    bc = c - b

    # 내적과 벡터의 크기를 이용하여 각도 계산
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    # 안정성을 위해 cos 값을 -1.0 ~ 1.0 범위로 제한 (부동소수점 오차 방지)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    # 라디안을 도로 변환
    return np.degrees(angle)

# 4. CSV 파일들을 순차적으로 처리
csv_files = [f for f in os.listdir(CSV_INPUT_DIR) if f.endswith('.csv')]

for csv_file in csv_files:
    input_path = os.path.join(CSV_INPUT_DIR, csv_file)
    output_path = os.path.join(PROCESSED_OUTPUT_DIR, os.path.splitext(csv_file)[0] + '_processed.csv')
    
    print(f"Processing CSV: {csv_file}")
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Skipping.")
        continue

    # 각 프레임(행)마다 고관절 각도 계산
    for index, row in df.iterrows():
        # 각 관절 부위별 좌표
        left_hip_coords = np.array([row[f'lm_{LANDMARKS["LEFT_HIP"]}_x'], row[f'lm_{LANDMARKS["LEFT_HIP"]}_y'], row[f'lm_{LANDMARKS["LEFT_HIP"]}_z']])
        left_knee_coords = np.array([row[f'lm_{LANDMARKS["LEFT_KNEE"]}_x'], row[f'lm_{LANDMARKS["LEFT_KNEE"]}_y'], row[f'lm_{LANDMARKS["LEFT_KNEE"]}_z']])
        left_shoulder_coords = np.array([row[f'lm_{LANDMARKS["LEFT_SHOULDER"]}_x'], row[f'lm_{LANDMARKS["LEFT_SHOULDER"]}_y'], row[f'lm_{LANDMARKS["LEFT_SHOULDER"]}_z']])
        left_ankle_coords = np.array([row[f'lm_{LANDMARKS["LEFT_ANKLE"]}_x'], row[f'lm_{LANDMARKS["LEFT_ANKLE"]}_y'], row[f'lm_{LANDMARKS["LEFT_ANKLE"]}_z']])

        right_hip_coords = np.array([row[f'lm_{LANDMARKS["RIGHT_HIP"]}_x'], row[f'lm_{LANDMARKS["RIGHT_HIP"]}_y'], row[f'lm_{LANDMARKS["RIGHT_HIP"]}_z']])
        right_knee_coords = np.array([row[f'lm_{LANDMARKS["RIGHT_KNEE"]}_x'], row[f'lm_{LANDMARKS["RIGHT_KNEE"]}_y'], row[f'lm_{LANDMARKS["RIGHT_KNEE"]}_z']])
        right_shoulder_coords = np.array([row[f'lm_{LANDMARKS["RIGHT_SHOULDER"]}_x'], row[f'lm_{LANDMARKS["RIGHT_SHOULDER"]}_y'], row[f'lm_{LANDMARKS["RIGHT_SHOULDER"]}_z']])
        right_ankle_coords = np.array([row[f'lm_{LANDMARKS["RIGHT_ANKLE"]}_x'], row[f'lm_{LANDMARKS["RIGHT_ANKLE"]}_y'], row[f'lm_{LANDMARKS["RIGHT_ANKLE"]}_z']])

        # 몸통의 중심선 기준점 계산 : 왼쪽 고관절과 오른쪽 고관절의 중점
        mid_hip_coords = (np.array(left_hip_coords) + np.array(right_hip_coords)) / 2
        
        # 1. 고관절 굴곡/신전 각도 계산(어깨, 고관절, 무릎을 사용하여 계산)
        df.loc[index, 'left_hip_flex_ext_angle'] = calculate_angle(left_shoulder_coords, left_hip_coords, left_knee_coords)
        df.loc[index, 'right_hip_flex_ext_angle'] = calculate_angle(right_shoulder_coords, right_hip_coords, right_knee_coords)

        # 2. 고관절 내전/외전 각도(몸통의 중심선, 고관절, 무릎을 사용하여 계산)
        df.loc[index, 'left_hip_abd_add_angle'] = calculate_angle(mid_hip_coords, left_hip_coords, left_knee_coords)
        df.loc[index, 'right_hip_abd_add_angle'] = calculate_angle(mid_hip_coords, right_hip_coords, right_knee_coords)

        # 3. 발레 특화 : 고관절 90도 굴곡 상태에서 고관절 외전&외회전의 복합적인 움직임 각도

        # 90도 정도 굴곡된 상태에서 무릎의 Z축 위치 변화를 이용해 회전 각도를 근사
        # 몸통 중심선을 기준으로 고관절-무릎 벡터가 Z축을 중심으로 얼마나 돌아갔는지 측정
        
        # 기준 평면 벡터: 고관절-중심선 벡터
        left_hip_to_mid = mid_hip_coords - left_hip_coords
        right_hip_to_mid = mid_hip_coords - right_hip_coords
        
        # 허벅지 벡터: 고관절-무릎 벡터
        left_hip_to_knee = left_knee_coords - left_hip_coords
        right_hip_to_knee = right_knee_coords - right_hip_coords

        # 두 벡터의 평면상(x, z축) 각도를 계산하여 회전 각도 근사
        left_angle_xz_plane = calculate_angle([left_hip_to_mid[0], 0, left_hip_to_mid[2]], [0, 0, 0], [left_hip_to_knee[0], 0, left_hip_to_knee[2]]) #y축(위아래)를 0으로 만들고, x축과 z축만 남김 -> x-z 평면으로 만듦.
        df.loc[index, 'left_hip_external_rotation_approx_angle'] = left_angle_xz_plane

        right_angle_xz_plane = calculate_angle([right_hip_to_mid[0], 0, right_hip_to_mid[2]], [0, 0, 0], [right_hip_to_knee[0], 0, right_hip_to_knee[2]]) #y축(위아래)를 0으로 만들고, x축과 z축만 남김 -> x-z 평면으로 만듦.
        df.loc[index, 'right_hip_external_rotation_approx_angle'] = right_angle_xz_plane

        # 4. 발레 특화 : 고관절 외전&굴곡&외회전의 복합적인 움직임 각도

        # 1) 고관절 외전/내전 각도 계산 (중점-고관절-무릎)
        abduction_angle = calculate_angle(mid_hip_coords, left_hip_coords, left_knee_coords)
        df.loc[index, 'left_hip_abduction_angle'] = abduction_angle

        # 2) 고관절 굴곡 각도 계산 (어깨-고관절-무릎)
        left_flexion_angle = calculate_angle(left_shoulder_coords, left_hip_coords, left_knee_coords)
        df.loc[index, 'left_hip_flexion_angle'] = left_flexion_angle

        # 3) 고관절의 외전&외회전 복합움직임 각도 계산(3과 같음)

    # 가공된 데이터 저장
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

print("All CSV files have been processed and saved.")
