\# 프로젝트: 컴퓨터 비전 기반 고관절 움직임 분석 (Computer Vision-based Hip Motion Analysis)



\## 📌 프로젝트 소개 (Introduction)



이 프로젝트는 컴퓨터 비전 기술을 활용하여 2D 영상 속 인체의 고관절 움직임을 정량적으로 분석합니다. 무용 동작과 같은 복합적인 움직임의 과학적 원리를 규명하고, 이를 통해 인체 역학에 대한 깊은 인사이트를 얻는 것을 목표로 합니다. 특히, 단순한 각도 측정을 넘어 여러 각도(굴곡, 외전, 외회전)가 복합적으로 발생하는 다면적인 움직임을 분석하는 데 중점을 두었습니다.



\## 🚀 주요 기능 (Key Features)



\* 실시간 포즈 추정 : 영상에서 MediaPipe 라이브러리를 사용해 인체 주요 랜드마크를 추출합니다.

\* 고관절 각도 계산 : 랜드마크 데이터를 기반으로 굴곡/신전, 외전/내전, 외회전/내회전 등 고관절의 다양한 각도를 계산합니다.

\* 영상-그래프 동기화 시각화 : 분석된 각도 데이터를 시간 흐름에 따라 그래프로 시각화하고, 이를 원본 영상과 동기화하여 동시에 재생합니다. 이를 통해 '어떤 동작을 할 때 각도가 어떻게 변하는지'를 직관적으로 파악할 수 있습니다.

\* 샘플 데이터 제공 : 분석된 데이터는 CSV 파일로 저장되어 제공됩니다.



\## 🛠️ 기술 스택 (Technologies Used)



\* 언어 : Python

\* 주요 라이브러리 : OpenCV, MediaPipe, NumPy, pandas, Matplotlib, SciPy



\## 📁 파일 구조 (File Structure)



\* 1 extract\_data.py : 영상에서 랜드마크를 추출하고 CSV 파일로 저장하는 코드입니다.

\* 2 calculate\_angle.py : 랜드마크 데이터(CSV)를 기반으로 고관절 각도를 계산하는 핵심 로직입니다.

\* 3 video\_and\_graph\_visualizer.py : 계산된 각도 데이터를 영상과 함께 동기화하여 시각화하는 코드입니다.

\* processed\_data/ : 분석에 사용된 샘플 CSV 데이터 파일이 포함되어 있습니다.



\## 🎬 시연 영상 (Demonstration)



링크 추가 예정



\## 🚀 프로젝트 실행 방법 (How to Run)



1\.  필요 라이브러리 설치 : 터미널에서 다음 명령어를 실행하여 필요한 라이브러리를 설치합니다.

&nbsp;   `pip install opencv-python mediapipe numpy pandas matplotlib scipy`

2\.  영상 및 CSV 파일 경로 설정 : 각 py 파일 내 `VIDEO\_DIR`, `PROCESSED\_DATA\_DIR` 등의 변수의 경로를 본인의 환경에 맞게 수정합니다.

3\.  각 단계 실행 :

&nbsp;   \* `1 extract\_data.py`를 실행하여 랜드마크 데이터를 추출합니다.

&nbsp;   \* `2 calculate\_angle.py`를 실행하여 각도를 계산합니다.

&nbsp;   \* `3 video\_and\_graph\_visualizer.py`를 실행하여 최종 결과물을 확인합니다.

