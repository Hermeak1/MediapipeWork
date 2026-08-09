# MediapipeWork

MediaPipe **Holistic**으로 춤 영상의 인체 자세를 추적하는 실험 프로젝트.

## Holistic vs Pose — 왜 Holistic을 썼나

MediaPipe에는 자세 추정 기법이 여럿 있는데, 대상의 움직임 성격에 따라 선택이 갈립니다.

| 기법 | 적합한 상황 |
| --- | --- |
| **Holistic** | 춤처럼 **빠르고 큰 움직임**. 얼굴·손·몸을 함께 추적 |
| **Pose** | 스쿼트, 요가처럼 **정적인 자세를 정확히** 잡아야 할 때 |

가져온 영상이 춤 영상이라 Pose로는 추적이 자주 끊겼고, Holistic으로 바꾼 뒤 안정적으로 따라붙었습니다.

## 구현 포인트

`medi_dance.py`에 들어간 처리들입니다.

- **EMA 평활** (`EMA_ALPHA = 0.35`) — 랜드마크 좌표에 지수평활을 걸어 프레임 간 떨림을 줄입니다
- **프레임 스킵** (`INFER_EVERY_N = 2`) — 두 프레임에 한 번만 추론해 부하를 절반으로 낮춥니다
- **다운스케일** (`DOWNSCALE = 0.6`) — 입력 해상도를 줄여 처리 속도를 올립니다
- **visibility 임계값** (`VIS_THRESH = 0.55`) — 신뢰도가 낮은 랜드마크는 그리지 않습니다

## 실행

```bash
pip install mediapipe opencv-python
python medi_dance.py
```

같은 폴더에 `dance.mp4`가 있어야 합니다. 다른 파일을 쓰려면 `cv2.VideoCapture('dance.mp4')` 부분을 바꾸세요.

## 개발 환경

- Python 3.9 (Anaconda 가상환경)
- mediapipe, opencv-python

## 라이선스

MIT — [LICENSE](LICENSE) 참고.
