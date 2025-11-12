import cv2, time, collections, math
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing  = mp.solutions.drawing_utils
POSE = mp.solutions.pose.PoseLandmark
DOWNSCALE = 0.6            
INFER_EVERY_N = 2          
VIS_THRESH = 0.55          # visibility 임계값
EMA_ALPHA = 0.35          

last_landmarks = {}        # {lm_id: (x,y,z,vis)}
smoothed_landmarks = {}    # EMA 결과 저장

def ema_smooth(lm_id, x, y, z, vis):
    """지수평활(EMA). 첫 관측은 그대로."""
    if lm_id not in smoothed_landmarks:
        smoothed_landmarks[lm_id] = (x, y, z, vis)
        return x, y, z, vis
    lx, ly, lz, lv = smoothed_landmarks[lm_id]
    sx = EMA_ALPHA * x + (1-EMA_ALPHA) * lx
    sy = EMA_ALPHA * y + (1-EMA_ALPHA) * ly
    sz = EMA_ALPHA * z + (1-EMA_ALPHA) * lz
    sv = EMA_ALPHA * vis + (1-EMA_ALPHA) * lv
    smoothed_landmarks[lm_id] = (sx, sy, sz, sv)
    return sx, sy, sz, sv

def to_pixels(lm, w, h):
    return lm.x * w, lm.y * h, lm.z, lm.visibility

cap = cv2.VideoCapture('dance.mp4')
assert cap.isOpened(), "영상 열기 실패"

with mp_holistic.Holistic(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    frame_id = 0
    last_result = None

    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]

        # 다운스케일 입력 생성
        scaled = cv2.resize(frame, (int(w*DOWNSCALE), int(h*DOWNSCALE)))
        rgb = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False

        run_infer = (frame_id % INFER_EVERY_N == 0)
        if run_infer:
            last_result = holistic.process(rgb)
        result = last_result  

        if result and result.pose_landmarks:
            for lm_id, lm in enumerate(result.pose_landmarks.landmark):
                x_px = (lm.x * scaled.shape[1]) / DOWNSCALE
                y_px = (lm.y * scaled.shape[0]) / DOWNSCALE
                z_rel = lm.z
                vis   = lm.visibility

                # visibility 낮으면 이전값 유지
                if vis < VIS_THRESH and lm_id in last_landmarks:
                    x_px, y_px, z_rel, vis = last_landmarks[lm_id]
                else:
                    x_px, y_px, z_rel, vis = ema_smooth(lm_id, x_px, y_px, z_rel, vis)

                last_landmarks[lm_id] = (x_px, y_px, z_rel, vis)

            mp_drawing.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

        # FPS 표시
        cv2.putText(frame, f'Frame: {frame_id}', (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        cv2.imshow("Holistic (smoothed)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

cap.release()
cv2.destroyAllWindows()
