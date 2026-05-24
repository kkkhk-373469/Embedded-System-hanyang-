from base_ctrl import BaseController
from jetcam.csi_camera import CSICamera

import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO


# =====================================
# 차량 / 카메라 연결
# =====================================
base = BaseController('/dev/ttyUSB0', 115200)

camera = CSICamera(
    capture_width=1280,
    capture_height=720,
    downsample=2,
    capture_fps=30
)


# =====================================
# Segmentation 모델 설정
# =====================================
SEG_MODEL_PATH = 'best_line.pt'
DEVICE = 0 if torch.cuda.is_available() else 'cpu'

line_model = YOLO(SEG_MODEL_PATH)

print("[INFO] Segmentation model loaded:", SEG_MODEL_PATH)
print("[INFO] Classes:", line_model.names)
print("[INFO] Device:", DEVICE)


# =====================================
# Segmentation 추론 파라미터
# =====================================
CONF_THRESHOLD = 0.40
IMGSZ = 640
MASK_THRESHOLD = 0.50

# mask 픽셀이 너무 적으면 미검출로 처리
MIN_MASK_PIXELS = 30

# False이면 검출된 점선 전체 mask의 x 평균을 사용.
# 커브에서 먼 점선까지 포함되어 흔들리면 True로 바꾸고 하단 영역만 사용.
USE_BOTTOM_ROI = False
ROI_TOP_RATIO = 0.55


# =====================================
# 주행 제어 파라미터
# =====================================
# 주행 테스트는 낮은 속도부터 시작
SPEED = 0.08

Kp = 1.20
MAX_STEER = 0.80
MAX_SPEED = 0.30
TURN_STRENGTH = 0.30

# 조향 방향이 반대로 나오면 -1.0으로 변경
STEERING_SIGN = 1.0

# 네 기존 차량 코드처럼 계산된 좌/우 명령을 바꿔서 전달
# 조향 방향이 이상하면 먼저 이 값을 확인
SWAP_MOTOR_CHANNELS = True


# =====================================
# 디버그 설정
# =====================================
SAVE_DEBUG_IMAGE = True
DEBUG_IMAGE_PATH = 'seg_line_debug.jpg'
DEBUG_SAVE_INTERVAL = 5


# =====================================
# 공통 함수
# =====================================
def clip(value, max_value):
    return max(min(value, max_value), -max_value)


def stop_vehicle():
    base.base_json_ctrl({
        'T': 1,
        'L': 0.0,
        'R': 0.0
    })


def send_vehicle_command(steering, speed):
    steering = clip(steering, MAX_STEER)

    l_calc = speed - steering * TURN_STRENGTH
    r_calc = speed + steering * TURN_STRENGTH

    l_calc = clip(l_calc, MAX_SPEED)
    r_calc = clip(r_calc, MAX_SPEED)

    if SWAP_MOTOR_CHANNELS:
        sent_l = r_calc
        sent_r = l_calc
    else:
        sent_l = l_calc
        sent_r = r_calc

    base.base_json_ctrl({
        'T': 1,
        'L': float(sent_l),
        'R': float(sent_r)
    })

    return l_calc, r_calc, sent_l, sent_r


def get_pred_x_from_segmentation(frame):
    """
    검출된 모든 segmentation mask를 하나로 합치고,
    mask에 포함된 모든 픽셀의 x 좌표 평균을 pred_x로 사용한다.
    """
    height, width = frame.shape[:2]

    results = line_model.predict(
        source=frame,
        conf=CONF_THRESHOLD,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=False
    )

    if not results or results[0].masks is None:
        return None, None, np.zeros((height, width), dtype=np.uint8), 0

    # (검출 개수, mask_height, mask_width)
    masks = results[0].masks.data.detach().cpu().numpy()

    # 여러 점선 조각으로 검출되더라도 전부 합쳐서 사용
    combined_mask_small = np.any(
        masks > MASK_THRESHOLD,
        axis=0
    ).astype(np.uint8)

    # 모델 mask 크기를 실제 카메라 frame 크기로 변환
    combined_mask = cv2.resize(
        combined_mask_small,
        (width, height),
        interpolation=cv2.INTER_NEAREST
    )

    if USE_BOTTOM_ROI:
        roi_top = int(height * ROI_TOP_RATIO)
        combined_mask[:roi_top, :] = 0

    ys, xs = np.where(combined_mask > 0)
    mask_pixel_count = len(xs)

    if mask_pixel_count < MIN_MASK_PIXELS:
        return None, None, combined_mask, mask_pixel_count

    # 핵심: 검출된 점선 영역의 x 좌표 평균
    pred_x = float(xs.mean())
    pred_y = float(ys.mean())  # 디버그 화면 표시용

    return pred_x, pred_y, combined_mask, mask_pixel_count


def make_debug_image(frame, combined_mask, pred_x, pred_y, steering, mask_pixels):
    debug = frame.copy()
    height, width = debug.shape[:2]
    center_x = width / 2

    if combined_mask is not None:
        mask_bool = combined_mask > 0
        overlay = debug.copy()
        overlay[mask_bool] = (0, 255, 0)
        debug = cv2.addWeighted(debug, 0.70, overlay, 0.30, 0)

    # 화면 중심
    cv2.line(
        debug,
        (int(center_x), 0),
        (int(center_x), height),
        (255, 0, 0),
        2
    )

    if pred_x is not None and pred_y is not None:
        # mask x 평균 위치
        cv2.circle(
            debug,
            (int(pred_x), int(pred_y)),
            10,
            (0, 0, 255),
            -1
        )

        cv2.line(
            debug,
            (int(center_x), int(pred_y)),
            (int(pred_x), int(pred_y)),
            (0, 255, 255),
            2
        )

        pred_text = f"PredX: {pred_x:.1f}"
    else:
        pred_text = "PredX: None / STOP"

    cv2.putText(
        debug,
        pred_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        debug,
        f"Steer: {steering:+.3f} MaskPx: {mask_pixels}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    return debug


# =====================================
# MAIN LOOP
# =====================================
frame_count = 0

try:
    print("[START] Segmentation line-following test")
    print("[SAFETY] 점선 mask 미검출 시 즉시 정지합니다.")

    while True:
        frame = camera.read()

        if frame is None:
            print("[ERROR] Cannot read camera frame -> STOP")
            stop_vehicle()
            time.sleep(0.1)
            continue

        height, width = frame.shape[:2]
        center_x = width / 2

        pred_x, pred_y, combined_mask, mask_pixels = (
            get_pred_x_from_segmentation(frame)
        )

        if pred_x is None:
            steering = 0.0
            stop_vehicle()

            print(
                f"[STOP] Line mask not detected "
                f"| mask_pixels={mask_pixels}"
            )

        else:
            error = pred_x - center_x
            error_norm = error / center_x

            steering = STEERING_SIGN * Kp * error_norm
            steering = clip(steering, MAX_STEER)

            l_calc, r_calc, sent_l, sent_r = send_vehicle_command(
                steering,
                SPEED
            )

            print(
                f"[AUTO] pred_x={pred_x:.1f} "
                f"center_x={center_x:.1f} "
                f"error={error:+.1f} "
                f"steer={steering:+.3f} "
                f"Sent_L={sent_l:+.3f} "
                f"Sent_R={sent_r:+.3f}"
            )

        if SAVE_DEBUG_IMAGE and frame_count % DEBUG_SAVE_INTERVAL == 0:
            debug_image = make_debug_image(
                frame,
                combined_mask,
                pred_x,
                pred_y,
                steering,
                mask_pixels
            )
            cv2.imwrite(DEBUG_IMAGE_PATH, debug_image)

        frame_count += 1
        time.sleep(0.03)

except KeyboardInterrupt:
    print("\n[STOP] KeyboardInterrupt")

finally:
    stop_vehicle()
    cv2.destroyAllWindows()
    print("[STOP] Vehicle stopped.")
