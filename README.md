# 지능형 실내 CCTV 영상 분석 시스템

> **스마트폰을 CCTV로 활용하여 실내 공간의 사람을 실시간 탐지·추적하고, 평면도에 위치를 자동으로 표시하는 AI 기반 영상 분석 시스템**

---

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [주요 기능](#주요-기능)
4. [기술 스택](#기술-스택)
5. [파일 구조](#파일-구조)
6. [설치 방법](#설치-방법)
7. [실행 방법](#실행-방법)
8. [카메라 추가 방법](#카메라-추가-방법)
9. [Homography 매핑 방법](#homography-매핑-방법)
10. [키 보정 방법](#키-보정-방법)
11. [웹 대시보드 사용법](#웹-대시보드-사용법)
12. [핵심 파라미터 설명](#핵심-파라미터-설명)
13. [알고리즘 설명](#알고리즘-설명)
14. [트러블슈팅](#트러블슈팅)

---

## 프로젝트 개요

### 개발 배경

미아 전단지나 범죄 용의자 수배 시 공통적으로 체격과 착의 사항이 핵심 식별 정보로 활용됩니다. 기존 CCTV 시스템은 관제 요원이 수동으로 영상을 모니터링해야 하는 한계가 있습니다. 본 시스템은 AI를 활용해 사람을 자동으로 탐지·추적하고, 성별·복장·키·체형 등의 특징을 자동 분류하여 실내 평면도에 실시간으로 위치를 표시합니다.

### 핵심 목표

- 스마트폰 카메라를 CCTV로 활용하여 별도 장비 없이 시스템 구축
- YOLOv8 + DeepSORT + OSNet 기반 실시간 다중 인물 탐지 및 추적
- 카메라 시야 밖으로 나갔다 돌아와도 동일 ID 유지 (외형 기반 Re-ID)
- 웹 대시보드에서 평면도 기반 실시간 위치 모니터링

---

## 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                   스마트폰 (CCTV)                        │
│  Safari/Chrome 브라우저 → 카메라 → JPEG 프레임 전송     │
└────────────────────────┬────────────────────────────────┘
                         │ WSS WebSocket (암호화)
                         ▼
┌─────────────────────────────────────────────────────────┐
│              stream_server.py (Python)                   │
│  - HTTPS 서버 (폰 접속용)                               │
│  - WSS WebSocket 서버 (프레임 수신)                     │
│  - 카메라 동적 추가 API (포트 8099)                     │
│  - 키 보정 API                                          │
│  - Homography 저장 API                                  │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP API
                         ▼
┌─────────────────────────────────────────────────────────┐
│           cctv_capstone.exe (C++17)                      │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  YOLOv8n    │  │  DeepSORT    │  │  OSNet Re-ID  │  │
│  │  사람 탐지  │→ │  객체 추적   │→ │  외형 특징    │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ MobileNetV2 │  │  Homography  │  │  MJPEG 서버   │  │
│  │ 복장 분류   │  │  좌표 변환   │  │  영상 스트림  │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│  - 카메라 관리 API (포트 8098)                          │
│  - positions.json 실시간 저장                           │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP
                         ▼
┌─────────────────────────────────────────────────────────┐
│              index.html (웹 대시보드)                    │
│  - Leaflet.js 평면도 마커 시각화                        │
│  - MJPEG 실시간 영상 표시                               │
│  - 카메라 추가 / Homography 매핑 UI                     │
│  - 인물 필터 검색 (성별, 복장, 키, 색상)               │
│  - 1/2/4분할 화면                                       │
└─────────────────────────────────────────────────────────┘
```

---

## 주요 기능

### 1. 실시간 사람 탐지
- YOLOv8n ONNX 모델 기반 CPU 실시간 추론
- Confidence 임계값: 0.25 / NMS IoU 임계값: 0.35
- 프레임당 평균 처리 시간: ~30~100ms (CPU 환경)

### 2. 다중 인물 추적 (DeepSORT + OSNet)
- 카메라 시야 밖으로 나갔다 돌아와도 동일 ID 유지
- 외형 기반 매칭 (위치 정보 미사용)
- 매칭 가중치 구성:
  | 특징 | 가중치 |
  |------|--------|
  | OSNet 외형 벡터 | 20% |
  | 성별 | 15% |
  | 상의 종류 | 15% |
  | 하의 종류 | 15% |
  | 체형 비율 | 15% |
  | 키 범주 | 10% |
  | 상의 색상 | 5% |
  | 하의 색상 | 5% |

### 3. 외형 특징 자동 분류
- **성별**: MobileNetV2 기반 분류 (male/female)
- **상의 종류**: tshirt / hoodie / jacket / shirt
- **하의 종류**: pants / skirt / shorts
- **색상**: HSV 히스토그램 기반 (red/orange/yellow/green/blue/purple/white/black/gray)
- **키**: 5단계 (~140cm / 140~150cm / 150~160cm / 160~170cm / 170cm~)
- **체형 비율**: 어깨너비 / 키 비율

### 4. 실내 위치 매핑 (Homography)
- 카메라 화면 4점 ↔ 평면도 4점 대응으로 변환 행렬 자동 계산
- 웹 UI에서 클릭만으로 간편 매핑
- 카메라별 독립적인 Homography 저장

### 5. 동적 카메라 관리
- 웹 대시보드에서 카메라 추가/제거
- 추가 시 자동으로 WSS/HTTPS/MJPEG 서버 생성
- 카메라 수 제한 없음
- 아이폰, 갤럭시, 갤럭시 탭 등 모든 스마트폰 지원

### 6. 다중 지도 관리
- 여러 평면도 등록 및 탭 전환
- 카메라별 담당 지도 지정

### 7. 웹 대시보드
- 1/2/4분할 화면 레이아웃
- 각 패널에서 카메라 영상 또는 평면도 선택
- 인물 필터 검색 (성별, 상의/하의 종류, 색상, 키)
- 실시간 탐지 인원 수 표시

---

## 기술 스택

| 구분 | 기술 |
|------|------|
| 언어 | C++17, Python 3.13, JavaScript |
| 빌드 | CMake 3.20+, Visual Studio 2022 |
| AI 탐지 | YOLOv8n (ONNX Runtime 1.24.4) |
| AI 추적 | DeepSORT (Kalman Filter + Hungarian Algorithm) |
| Re-ID | OSNet (ONNX) |
| 복장 분류 | MobileNetV2 (ONNX) |
| 컴퓨터 비전 | OpenCV 4.8.0 |
| 선형대수 | Eigen3 |
| 웹 스트리밍 | Python HTTPS/WSS, C++ Winsock2 MJPEG |
| 지도 시각화 | Leaflet.js 1.9.4 |
| 데이터 포맷 | JSON (nlohmann/json) |
| OS | Windows 11 |

---

## 파일 구조

```
C:\cctvcapstone\
├── start.bat                    # 전체 시스템 시작 (더블클릭)
├── stop.bat                     # 전체 시스템 종료
├── stream_server.py             # 스마트폰 카메라 스트리밍 서버
├── index.html                   # 웹 대시보드
├── positions.json               # 실시간 탐지 결과
├── cert.pem / key.pem           # SSL 인증서 (HTTPS용)
├── osnet.onnx                   # Re-ID 모델
├── clothing.onnx                # 복장 분류 모델
├── homography.yml               # 기본 Homography 행렬
├── homography_cam1.yml          # CAM1 Homography (매핑 후 생성)
├── homography_cam2.yml          # CAM2 Homography (매핑 후 생성)
├── maps\
│   ├── map_1.png                # 지도 1 평면도 이미지
│   └── map_2.png                # 지도 2 평면도 이미지 (추가 시)
├── opencv\                      # OpenCV 4.8.0 라이브러리
├── onnxruntime\                 # ONNX Runtime 1.24.4 라이브러리
├── eigen\                       # Eigen3 라이브러리
├── vcpkg\                       # vcpkg 패키지 매니저
└── project\cctv_capstone\
    ├── main.cpp                 # C++ 메인 (탐지/추적/스트리밍)
    ├── calibration.cpp          # Homography 캘리브레이션 도구
    ├── yolov8n.onnx             # YOLOv8n 모델
    ├── json.hpp                 # JSON 라이브러리
    ├── CMakeLists.txt           # CMake 빌드 설정
    └── deepsort\
        ├── kalman_filter.h/cpp  # 칼만 필터
        ├── hungarian.h/cpp      # 헝가리안 알고리즘
        ├── track.h/cpp          # 트랙 관리
        └── tracker.h/cpp        # DeepSORT 트래커
```

---

## 설치 방법

### 사전 요구사항

- Windows 11
- Visual Studio 2022 (C++ 데스크톱 개발 워크로드)
- CMake 3.20 이상
- Python 3.13 이상
- Git

### 1. 저장소 클론

```bash
git clone https://github.com/LIMJUNYOUNG1120/cctv_capstone.git C:\cctvcapstone\project\cctv_capstone
```

### 2. 라이브러리 설치

**OpenCV 4.8.0**
```
https://opencv.org/releases/ 에서 Windows 버전 다운로드
C:\cctvcapstone\opencv\ 에 설치
```

**ONNX Runtime 1.24.4**
```
https://github.com/microsoft/onnxruntime/releases 에서
onnxruntime-win-x64-gpu-1.24.4 다운로드
C:\cctvcapstone\onnxruntime\lib\ 에 압축 해제
```

**Eigen3**
```
https://eigen.tuxfamily.org 에서 다운로드
C:\cctvcapstone\eigen\ 에 압축 해제
```

**Python 패키지**
```bash
pip install websockets opencv-python numpy
```

### 3. SSL 인증서 생성

```bash
cd C:\cctvcapstone
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=172.20.10.3"
```

> ⚠️ IP 주소(172.20.10.3)를 본인 PC의 IP로 변경해주세요.

### 4. stream_server.py IP 설정

`stream_server.py` 파일 상단의 IP를 본인 PC IP로 변경:

```python
IP = "172.20.10.3"  # ← 본인 PC의 IP로 변경
```

### 5. 빌드

```bash
cd C:\cctvcapstone\project\cctv_capstone
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

# DLL 복사
copy C:\cctvcapstone\opencv\opencv\build\x64\vc16\bin\opencv_world480.dll Release\
copy C:\cctvcapstone\onnxruntime\lib\onnxruntime-win-x64-gpu-1.24.4\lib\onnxruntime.dll Release\
```

---

## 실행 방법

### 간편 실행 (권장)

`C:\cctvcapstone\start.bat` 을 **우클릭 → 관리자로 실행**

> 처음 한 번만 관리자로 실행하면 이후에는 더블클릭으로 실행 가능합니다.

자동으로 아래가 실행됩니다:
1. 방화벽 포트 오픈
2. stream_server.py 시작
3. cctv_capstone.exe 시작
4. 웹서버 시작
5. 브라우저 자동 오픈

### 수동 실행

```bash
# 터미널 1
cd C:\cctvcapstone
python stream_server.py

# 터미널 2
cd C:\cctvcapstone\project\cctv_capstone\build
.\Release\cctv_capstone.exe

# 터미널 3
cd C:\cctvcapstone
python -m http.server 8000
```

브라우저에서 접속:
```
http://{PC_IP}:8000/index.html
```

### 종료

`C:\cctvcapstone\stop.bat` 더블클릭

---

## 카메라 추가 방법

1. 웹 대시보드 오른쪽 상단 **+ 카메라 추가** 버튼 클릭
2. 카메라 정보 입력:
   - **카메라 이름**: CAM 1, CAM 2 등
   - **기준 사람 키**: 키 보정에 사용할 기준값 (cm)
   - **담당 지도**: 이 카메라의 위치를 표시할 지도 선택
3. **추가** 버튼 클릭
4. 2단계 모달에서 접속 주소 확인
5. 스마트폰 브라우저(Safari/Chrome)에서 접속 주소 입력
   - 예: `https://172.20.10.3:8080`
   - 보안 경고 뜨면 **고급 → 안전하지 않은 사이트로 이동** 클릭
6. **카메라 시작** 버튼 클릭
7. 카메라 프리뷰 확인 후 키 보정 진행

> ✅ 스마트폰과 PC가 같은 WiFi 또는 핫스팟에 연결되어 있어야 합니다.

---

## Homography 매핑 방법

카메라 화면의 좌표를 평면도 좌표로 변환하기 위한 설정입니다. 카메라 추가 후 키 보정 완료 시 자동으로 매핑 화면이 열립니다.

### 매핑 순서

1. 매핑 모달에서 **카메라 화면(왼쪽)** 과 **지도 화면(오른쪽)** 이 표시됨
2. **카메라 화면에서 바닥의 특정 지점 클릭** (오렌지색 점 표시)
3. **지도 화면에서 같은 지점의 위치 클릭** (초록색 점 표시)
4. 위 과정을 4번 반복 (총 4쌍의 대응점 선택)
5. **매핑 완료** 버튼 클릭

### 좋은 매핑을 위한 팁

```
✅ 4개의 점이 넓게 분산되어야 정확도가 높아요
✅ 바닥에 있는 점을 선택해야 해요 (사람 발 위치 기준)
✅ 카메라 화면 4개 코너 근처 점을 선택하면 좋아요
❌ 4개 점이 일직선 위에 있으면 안 돼요
```

---

## 키 보정 방법

카메라 설치 높이와 각도에 따라 키 추정 오차가 발생합니다. 기준 키를 아는 사람이 카메라 앞에 서면 자동으로 보정 계수를 계산합니다.

### 보정 순서

1. 카메라 추가 2단계 모달에서 **기준 키** 확인 (예: 170cm)
2. 해당 키의 사람이 카메라 화면 중앙에 서기
3. **키 보정 시작** 버튼 클릭
4. 3초간 가만히 서 있기
5. **보정 완료** 메시지 확인

> 보정 없이도 동작하지만 오차가 ±20cm 이상 발생할 수 있습니다.

---

## 웹 대시보드 사용법

### 화면 구성

```
┌─────────────────────────────────────────────────────┐
│ 🎥 지능형 영상 분석 시스템  탐지:5명 카메라:2대 +추가│
│ [1분할][2분할][4분할] 성별▼ 상의▼ 하의▼ 색상▼ 키▼  │
│                      [검색] [초기화]                 │
├─────────────────────┬───────────────────────────────┤
│  📷 카메라  CAM1▼   │  📷 카메라  🗺️지도▼           │
│                     │  ┌──────┐  ┌──────┐           │
│  [MJPEG 영상]       │  │지도1 │  │+추가 │           │
│                     │  └──────┘  └──────┘           │
│                     │  [평면도 + 마커]               │
└─────────────────────┴───────────────────────────────┘
```

### 주요 기능

| 기능 | 설명 |
|------|------|
| 화면 분할 | 1/2/4분할 선택 가능 |
| 패널 선택 | 각 패널에서 카메라 영상 또는 지도 선택 |
| 인물 필터 | 성별, 상의/하의 종류, 색상, 키로 검색 |
| 지도 탭 | 여러 평면도 등록 및 전환 |
| 마커 툴팁 | 마커 클릭 시 인물 상세 정보 표시 |

### 마커 색상 의미

```
마커 상단: 상의 색상
마커 하단: 하의 색상
마커 번호: 인물 ID
```

---

## 핵심 파라미터 설명

### YOLOv8 탐지 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| Confidence | 0.25 | 이 값 이상의 확신도일 때만 탐지 |
| NMS IoU | 0.35 | 겹치는 박스 제거 기준 |

### DeepSORT 추적 파라미터

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| maxMisses | 60 | 화면 밖으로 나간 후 ID를 유지하는 프레임 수 |
| minHits | 2 | ID 부여 전 최소 탐지 횟수 |
| matchThreshold | 0.6 | 60% 이상 유사할 때 동일 인물로 판단 |
| featureUpdateInterval | 3 | 외형 특징 업데이트 주기 (프레임) |

---

## 알고리즘 설명

### YOLOv8n
You Only Look Once v8 nano 모델. 이미지를 한 번만 보고 객체의 위치와 종류를 동시에 예측합니다. nano 버전은 경량화 모델로 CPU에서도 실시간 처리가 가능합니다.

### DeepSORT
Simple Online and Realtime Tracking with a Deep association metric. 칼만 필터로 다음 위치를 예측하고, 헝가리안 알고리즘으로 이전 트랙과 새 탐지 결과를 매칭합니다. 본 시스템에서는 위치 정보 없이 외형 특징만으로 매칭하여 카메라 시야 밖에서 돌아온 인물도 동일 ID를 유지합니다.

### OSNet (Omni-Scale Network)
다양한 스케일의 특징을 동시에 추출하는 Re-ID 전용 네트워크. 같은 사람을 다른 각도나 조명에서도 동일인으로 인식하는 데 특화되어 있습니다.

### Homography
두 평면(카메라 화면과 실내 평면도) 사이의 투영 변환 행렬. 카메라 화면상의 발 위치 픽셀 좌표를 평면도상의 실제 위치로 변환합니다. 4쌍의 대응점으로 3×3 변환 행렬을 계산합니다.

---

## 트러블슈팅

### 스마트폰에서 카메라가 실행되지 않아요
```
원인: HTTPS 연결 필요
해결: cert.pem, key.pem이 C:\cctvcapstone\ 에 있는지 확인
     접속 주소가 https:// 로 시작하는지 확인
     보안 경고 시 "고급 → 안전하지 않은 사이트로 이동" 클릭
```

### 사이트에 연결할 수 없어요
```
원인: 방화벽 차단 또는 서버 미실행
해결: start.bat을 관리자 권한으로 실행
     PC와 스마트폰이 같은 네트워크인지 확인
     ipconfig로 PC IP 확인 후 stream_server.py의 IP와 일치하는지 확인
```

### 탐지가 잘 안 돼요
```
원인: 조명 부족, Confidence 임계값 문제
해결: 조명 밝기 확인
     main.cpp에서 if (personScore < 0.25f) 를 0.2f로 낮춰 재빌드
```

### ID가 자꾸 바뀌어요
```
원인: matchThreshold가 너무 높거나 외형 특징 추출 오류
해결: tracker.cpp에서 matchThreshold_ 값을 0.5로 낮추기
     카메라 해상도와 조명 상태 확인
```

### 평면도 위치가 실제와 달라요
```
원인: Homography 매핑 정확도 부족
해결: 4개 점을 화면 코너 근처에 넓게 분산하여 다시 매핑
     바닥에 있는 점을 선택했는지 확인
     평면도가 실제 비율대로 그려진 도면인지 확인
```

### 빌드 오류가 발생해요
```
원인: 라이브러리 경로 문제
해결: CMakeLists.txt의 경로가 실제 설치 경로와 일치하는지 확인
     Visual Studio 2022의 C++ 데스크톱 개발 워크로드가 설치되어 있는지 확인
```

---

## 포트 구성

| 포트 | 용도 |
|------|------|
| 8000 | 웹 대시보드 HTTP 서버 |
| 8080, 8082, 8083... | 카메라별 HTTPS 서버 (폰 접속용) |
| 8091, 8092, 8093... | 카메라별 MJPEG 스트림 서버 |
| 8098 | cctv_capstone.exe 카메라 관리 API |
| 8099 | stream_server.py 카메라 관리 API |
| 8765, 8766, 8767... | 카메라별 WSS WebSocket 서버 |

---

## GitHub

```
https://github.com/LIMJUNYOUNG1120/cctv_capstone
```

---

## 개발 환경

```
OS:      Windows 11
IDE:     Visual Studio Code, Visual Studio 2022
언어:    C++17, Python 3.13
빌드:    CMake 4.3.2
GPU:     CPU 모드 (CUDA 미사용)
테스트:  노트북 내장 웹캠, iPhone
```

---

*본 시스템은 캡스톤 디자인 프로젝트로 개발되었습니다.*
