# face.py  (YOLO + hotspot blob fallback + mouse pixel temperature)
import time
import cv2
import numpy as np
from ultralytics import YOLO

from read_frame import get_frame  # rgb_frame, raw_frame 반환

# YOLO 일반 모델 (COCO, class 0 = person)
model = YOLO("yolov8n.pt")

# Lepton 해상도
WIDTH = 160
HEIGHT = 120

# YOLO 감지 주기 (초) - 1초에 한 번만 사람 detection
DETECTION_INTERVAL = 1.0

last_det_time = 0.0
person_box = None          # (x1, y1, x2, y2)
head_temp_c = None         # 마지막 추정 머리 온도(섭씨)
det_source = None          # "YOLO" or "BLOB"

# ===== Mouse Temperature Reader =====
mouse_x, mouse_y = -1, -1

def mouse_event(event, x, y, flags, param):
    """
    OpenCV 창 위에서 마우스를 움직일 때
    현재 (x, y) 좌표를 저장해두고,
    main loop에서 raw_frame[y, x] 읽어서 온도 표시
    """
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y


def raw_to_celsius(raw_val: float) -> float:
    """
    Radiometric Lepton 3.x 가정:
    raw ≈ Kelvin * 100  =>  T(°C) = raw/100 - 273.15
    절대 정확도는 캘리브레이션 필요하지만
    상대적인 비교/제어 용도로는 충분함.
    """
    return raw_val * 0.01 - 273.15


def find_head_hotspot(raw_frame, box, head_ratio=0.6, radius=3):
    """
    사람 박스의 상단 부분(head 영역)에서 가장 뜨거운 픽셀(hotspot)을 찾고,
    그 주변 작은 영역 평균으로 머리 온도를 추정.

    raw_frame : (H, W) uint16 Lepton RAW
    box       : (x1, y1, x2, y2) - YOLO/BLOB가 찾은 사람 bbox
    head_ratio: 박스 상단에서 어느 비율까지를 머리 영역으로 볼지 (0~1)
    radius    : hotspot 주변 평균을 낼 반경 (픽셀)
    """
    x1, y1, x2, y2 = box

    # 이미지 범위 안으로 클램핑
    x1 = max(0, min(WIDTH - 1, int(x1)))
    x2 = max(0, min(WIDTH - 1, int(x2)))
    y1 = max(0, min(HEIGHT - 1, int(y1)))
    y2 = max(0, min(HEIGHT - 1, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    h = y2 - y1
    head_y2 = y1 + int(h * head_ratio)
    if head_y2 <= y1:
        head_y2 = y2

    roi = raw_frame[y1:head_y2, x1:x2]
    if roi.size == 0:
        return None

    # 0 값(유효하지 않은 픽셀) 제외
    mask = roi > 0
    if not mask.any():
        return None

    masked = np.where(mask, roi, 0)
    max_idx = np.argmax(masked)
    ry, rx = divmod(max_idx, roi.shape[1])

    hot_x = x1 + rx
    hot_y = y1 + ry

    # 주변 radius 내 픽셀 평균으로 노이즈 줄이기
    x_min = max(0, hot_x - radius)
    x_max = min(WIDTH, hot_x + radius + 1)
    y_min = max(0, hot_y - radius)
    y_max = min(HEIGHT, hot_y + radius + 1)

    neigh = raw_frame[y_min:y_max, x_min:x_max]
    valid = neigh[neigh > 0]
    if valid.size == 0:
        return None

    avg_raw = float(valid.mean())
    return raw_to_celsius(avg_raw)


def detect_blob_person_box(raw_frame, min_area=20):
    """
    YOLO가 아무도 못 잡았을 때 fallback:
    열화상 RAW에서 가장 뜨거운 blob(덩어리)을 찾아서 그 bounding box 반환.

    raw_frame : (H, W) uint16
    min_area  : 너무 작은 노이즈 blob 무시하기 위한 최소 면적
    """
    # 유효 영역(0이 아닌 값)만 사용
    valid = raw_frame[raw_frame > 0]
    if valid.size == 0:
        return None

    # 상위 90퍼센타일 이상을 "뜨거운 영역"으로 간주
    thr = np.percentile(valid, 90)
    mask = np.zeros_like(raw_frame, dtype=np.uint8)
    mask[raw_frame >= thr] = 255

    # binary mask에서 contour 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 가장 큰 blob 선택
    best_cnt = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > best_area:
            best_area = area
            best_cnt = cnt

    if best_cnt is None or best_area < min_area:
        return None

    x, y, w, h = cv2.boundingRect(best_cnt)
    return (x, y, x + w, y + h)


def main():
    global last_det_time, person_box, head_temp_c, det_source

    window_name = "Person Hotspot Temperature"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_event)

    while True:
        # 1) C++가 /dev/shm에 써둔 프레임 읽기
        rgb_frame, raw_frame = get_frame()  # rgb: (H, W, 3), raw: (H, W)

        now = time.time()

        # 2) 1초에 한 번만 YOLO + BLOB detection
        if now - last_det_time > DETECTION_INTERVAL:
            det_source = None
            person_box = None
            head_temp_c = None

            # ---- 2-1) 먼저 YOLO로 사람 탐지 시도 ----
            results = model(rgb_frame, imgsz=160, conf=0.25)

            yolo_box = None
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                if boxes.xyxy is not None and boxes.cls is not None:
                    xyxy = boxes.xyxy.cpu().numpy()
                    cls  = boxes.cls.cpu().numpy()

                    # class 0 = person
                    person_idx = np.where(cls == 0)[0]
                    if person_idx.size > 0:
                        # 여러 명이면 가장 큰 박스 선택
                        p_xyxy = xyxy[person_idx]
                        areas = (p_xyxy[:, 2] - p_xyxy[:, 0]) * (p_xyxy[:, 3] - p_xyxy[:, 1])
                        idx = np.argmax(areas)
                        x1, y1, x2, y2 = p_xyxy[idx]
                        yolo_box = (int(x1), int(y1), int(x2), int(y2))

            if yolo_box is not None:
                # YOLO로 사람 잘 잡은 경우
                person_box = yolo_box
                det_source = "YOLO"
                head_temp_c = find_head_hotspot(raw_frame, person_box,
                                                head_ratio=0.6, radius=3)
            else:
                # ---- 2-2) YOLO로 아무도 못 잡으면 BLOB fallback ----
                blob_box = detect_blob_person_box(raw_frame, min_area=20)
                if blob_box is not None:
                    person_box = blob_box
                    det_source = "BLOB"
                    # blob 박스 전체를 사람 덩어리로 보고 상단 60%를 머리 영역처럼 취급
                    head_temp_c = find_head_hotspot(raw_frame, person_box,
                                                    head_ratio=0.6, radius=3)
                else:
                    person_box = None
                    head_temp_c = None
                    det_source = None

            last_det_time = now

        # 3) 시각화용 이미지 복사
        vis = rgb_frame.copy()

        # 3-1) 마우스 아래 픽셀 온도 표시
        if 0 <= mouse_x < WIDTH and 0 <= mouse_y < HEIGHT:
            raw_val = int(raw_frame[mouse_y, mouse_x])
            if raw_val > 0:
                temp_c = raw_to_celsius(raw_val)
                txt = f"{temp_c:.2f} C"
                cv2.putText(vis, txt, (mouse_x + 8, mouse_y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.circle(vis, (mouse_x, mouse_y), 2, (255, 255, 255), -1)

        # 3-2) 사람 박스 + 머리 온도 + 탐지 소스 표시
        if person_box is not None:
            x1, y1, x2, y2 = person_box

            # YOLO면 초록, BLOB면 파랑 박스
            color = (0, 255, 0) if det_source == "YOLO" else (255, 0, 0)
            label = "Person(YOLO)" if det_source == "YOLO" else "Person(BLOB)"

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # 라벨 + 온도 텍스트
            if head_temp_c is not None:
                txt = f"{label} {head_temp_c:.1f} C"
            else:
                txt = label
            cv2.putText(vis, txt, (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # 4) 출력
        cv2.imshow(window_name, vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
