from djitellopy import Tello
import cv2
import numpy as np
import time
import os
import logging
from datetime import datetime


PHOTO_DIR = "photos"
# 若畫面顏色正常可改成 False；若黃變藍請維持 True
FRAME_IS_RGB = True
TARGET_MARKER_IDS = [0, 4, 5, 1, 2, 6, 7, 3]
TRANSITION_MODES = ["DOWN", "NEXT", "UP", "NEXT", "DOWN", "NEXT", "UP"]
USE_POSE_CONTROL = True
SEARCH_YAW_SPEED = -20
NEXT_FACE_RIGHT_SPEED = 18
NEXT_FACE_LEFT_YAW_SPEED = -18
SAME_FACE_DOWN_SPEED = -18
SAME_FACE_UP_SPEED = 18

# 姿態估計與閉迴路控制參數。
DISPLAY_WIDTH = 360
DISPLAY_HEIGHT = 240
FOCAL_LENGTH_PX = 330.0
MARKER_LENGTH_M = 0.10  # ArUco marker 實際邊長，單位: 公尺。
ANGLE_LOCK_TOLERANCE_DEG = 5.0  # yaw 小於此值才視為正對 marker。
XY_LOCK_TOLERANCE = 35  # 準心中心誤差小於此值才視為對準 marker。
X_EDGE_GUARD = 125  # 左右只做防丟失保護，不強迫完全置中。
MARKER_YAW_LR_SIGN = -1  # 若自動側滑方向相反，改成 -1。
PITCH_EDGE_SIGN = 1.0  # 若 pitch 正負方向和你的直覺相反，改成 -1.0。
SHOW_POSE_AXES = False  # OpenCV 藍色 Z 軸對平面 marker 容易翻面，測試時預設關閉。
TARGET_X_CM = 0.0
TARGET_Y_CM = 0.0
TARGET_Z_CM = 80.0
POSE_X_TOLERANCE_CM = 8.0
POSE_Y_TOLERANCE_CM = 8.0
POSE_Z_TOLERANCE_CM = 10.0
POSE_YAW_TOLERANCE_DEG = 7.0
MAX_AUTO_SPEED = 18
MIN_AUTO_SPEED = 8
POSE_LR_SIGN = 1
POSE_FB_SIGN = 1
POSE_UD_SIGN = -1
POSE_YAW_SIGN = 1

CAMERA_MATRIX = np.array(
    [
        [FOCAL_LENGTH_PX, 0.0, DISPLAY_WIDTH / 2.0],
        [0.0, FOCAL_LENGTH_PX, DISPLAY_HEIGHT / 2.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)


class PIDController:
    def __init__(self, kp, ki=0.0, kd=0.0, output_limit=MAX_AUTO_SPEED):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral = 0.0
        self.previous_error = None

    def reset(self):
        self.integral = 0.0
        self.previous_error = None

    def update(self, error, dt):
        if dt <= 0.0:
            dt = 1e-3

        self.integral += error * dt
        derivative = 0.0
        if self.previous_error is not None:
            derivative = (error - self.previous_error) / dt
        self.previous_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(np.clip(output, -self.output_limit, self.output_limit))


def _rc_speed(value, min_speed=MIN_AUTO_SPEED):
    speed = int(round(value))
    if speed == 0:
        return 0

    if abs(speed) < min_speed:
        speed = min_speed if speed > 0 else -min_speed

    return int(np.clip(speed, -MAX_AUTO_SPEED, MAX_AUTO_SPEED))


def _low_pass(previous, current, alpha=0.25):
    if current is None:
        return previous
    if previous is None:
        return float(current)
    return (1.0 - alpha) * previous + alpha * float(current)


def _normalize_angle_degrees(angle):
    def normalize(angle):
        if angle > 180.0:
            return angle - 360.0
        if angle < -180.0:
            return angle + 360.0
        return angle

    return float(normalize(angle))


def _rotation_matrix_to_yaw_degrees(rotation_matrix):
    # OpenCV 的 marker 座標系在正對時常落在 +/-180 度附近，因此轉成正對 = 0。
    raw_yaw = np.degrees(np.arctan2(rotation_matrix[0, 2], rotation_matrix[2, 2]))
    return _normalize_angle_degrees(raw_yaw - 180.0)


def _estimate_pitch_from_marker_edges(first_corners):
    # solvePnP 對平面 marker 的 pitch 容易上下翻面；先用上下邊長透視差做穩定估計。
    top_len = np.linalg.norm(first_corners[1] - first_corners[0])
    bottom_len = np.linalg.norm(first_corners[2] - first_corners[3])
    edge_sum = top_len + bottom_len

    if edge_sum < 1e-6:
        return None

    perspective_ratio = (bottom_len - top_len) / edge_sum
    pitch = np.degrees(np.arctan(2.5 * perspective_ratio))
    return float(np.clip(PITCH_EDGE_SIGN * pitch, -89.0, 89.0))


def _detect_aruco_detail(img, target_id=None):
    # 轉灰階可減少計算量，提升 ArUco 偵測速度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用指定字典 DICT_4X4_50
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    # 相容不同 OpenCV 版本:
    # 新版優先使用 ArucoDetector，舊版回退到 detectMarkers
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # 若沒有偵測到 marker
    if ids is None or len(ids) == 0:
        return None, None, None, None, None, None, None, None, None

    # 在顯示畫面上畫出 marker 框線
    cv2.aruco.drawDetectedMarkers(img, corners, ids)

    # 多個 marker 同時出現時，優先挑目前要拍的 target_id。
    marker_index = 0
    if target_id is not None:
        flat_ids = ids.flatten()
        matches = np.where(flat_ids == target_id)[0]
        if len(matches) == 0:
            return None, None, None, None, None, None, None, None, None
        marker_index = int(matches[0])

    first_corners = corners[marker_index][0]  # shape: (4, 2)
    marker_id = int(ids[marker_index][0])

    # marker 中心點 = 四個角點座標平均值
    cx = int(first_corners[:, 0].mean())
    cy = int(first_corners[:, 1].mean())

    # 畫面中心點
    h, w = img.shape[:2]
    frame_cx = w // 2
    frame_cy = h // 2

    # 誤差 = marker 中心點 - 畫面中心點
    error_x = cx - frame_cx
    error_y = cy - frame_cy

    # 使用第一個 marker 面積當作前後距離近似
    marker_area = int(cv2.contourArea(first_corners.astype("float32")))

    marker_yaw = None
    marker_pitch = _estimate_pitch_from_marker_edges(first_corners)
    pose_x_cm = None
    pose_y_cm = None
    pose_z_cm = None
    half_len = MARKER_LENGTH_M / 2.0
    object_points = np.array(
        [
            [-half_len, half_len, 0.0],
            [half_len, half_len, 0.0],
            [half_len, -half_len, 0.0],
            [-half_len, -half_len, 0.0],
        ],
        dtype=np.float32,
    )

    try:
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            first_corners.astype(np.float32),
            CAMERA_MATRIX,
            DIST_COEFFS,
            flags=getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE),
        )

        if success:
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            marker_yaw = _rotation_matrix_to_yaw_degrees(rotation_matrix)
            pose_x_cm = float(tvec[0][0] * 100.0)
            pose_y_cm = float(tvec[1][0] * 100.0)
            pose_z_cm = float(tvec[2][0] * 100.0)

            if SHOW_POSE_AXES and hasattr(cv2, "drawFrameAxes"):
                cv2.drawFrameAxes(
                    img,
                    CAMERA_MATRIX,
                    DIST_COEFFS,
                    rvec,
                    tvec,
                    MARKER_LENGTH_M * 0.5,
                )
    except cv2.error:
        marker_yaw = None
        pose_x_cm = None
        pose_y_cm = None
        pose_z_cm = None

    return (
        error_x,
        error_y,
        marker_id,
        marker_area,
        marker_yaw,
        marker_pitch,
        pose_x_cm,
        pose_y_cm,
        pose_z_cm,
    )


def find_aruco(img):
    # 依需求提供的基礎函式：只回傳 error_x, error_y, marker_id
    # 內部仍沿用同一套偵測流程與畫框邏輯
    error_x, error_y, marker_id, _, _, _, _, _, _ = _detect_aruco_detail(img)
    if marker_id is None:
        return None, None, None
    return error_x, error_y, marker_id


def reset_tracker_state(state):
    state["filtered_area"] = None
    state["filtered_ex"] = 0.0
    state["filtered_ey"] = 0.0
    state["filtered_x_cm"] = None
    state["filtered_y_cm"] = None
    state["filtered_z_cm"] = None
    state["filtered_yaw_deg"] = None
    state["fb_hold"] = True
    state["last_cmd"] = (0, 0, 0, 0)
    state["lock_start_ts"] = None
    state["last_pid_ts"] = None

    for pid in state.get("pids", {}).values():
        pid.reset()


def track_marker(tello, error_x, error_y):
    # Deadband：誤差落在容許範圍內時不動，降低震盪
    tolerance = 40
    speed = 15

    left_right_velocity = 0
    forward_backward_velocity = 0
    up_down_velocity = 0
    yaw_velocity = 0

    # 左右控制：error_x > 0 代表 marker 在右邊，無人機向右平移
    if error_x > tolerance:
        left_right_velocity = speed
    elif error_x < -tolerance:
        left_right_velocity = -speed

    # 上下控制：error_y > 0 代表 marker 在下方，無人機需下降
    if error_y > tolerance:
        up_down_velocity = -speed
    elif error_y < -tolerance:
        up_down_velocity = speed

    # 前後與旋轉暫時固定為 0
    tello.send_rc_control(
        left_right_velocity,
        forward_backward_velocity,
        up_down_velocity,
        yaw_velocity,
    )


def track_marker_with_distance_stable(
    tello,
    error_x,
    error_y,
    marker_area,
    marker_yaw,
    state,
):
    # X/Y 方向 deadband，減少小誤差抖動
    xy_tolerance = XY_LOCK_TOLERANCE

    # 前後距離控制: marker 面積越大代表越近
    target_area = 9000  # 目標面積，請依實際測試調整到合適值；過大過小都會影響穩定性。

    # 前後距離遲滯:
    # filtered_area 接近 target_area 才 hold；差太多就重新前後移動。
    dist_hold_tolerance = 200
    dist_release_tolerance = 350

    # 平滑量測
    alpha = 0.25
    if state["filtered_area"] is None:
        state["filtered_area"] = float(marker_area)
    else:
        state["filtered_area"] = (
            (1.0 - alpha) * state["filtered_area"] + alpha * float(marker_area)
        )
    state["filtered_ex"] = (1.0 - alpha) * state["filtered_ex"] + alpha * float(error_x)
    state["filtered_ey"] = (1.0 - alpha) * state["filtered_ey"] + alpha * float(error_y)

    left_right_velocity = 0
    forward_backward_velocity = 0
    up_down_velocity = 0
    yaw_velocity = 0

    ex = state["filtered_ex"]
    ey = state["filtered_ey"]

    # 水平置中: error_x 控制機身旋轉，讓 marker 留在準心上。
    if ex > xy_tolerance:
        yaw_velocity = 12 if ex < 90 else 18
    elif ex < -xy_tolerance:
        yaw_velocity = -12 if ex > -90 else -18

    # 正對 marker: marker_yaw 控制左右側滑，移到 marker 的正前方。
    if marker_yaw is not None and abs(marker_yaw) > ANGLE_LOCK_TOLERANCE_DEG:
        lr_speed = 10 if abs(marker_yaw) < 18 else 16
        left_right_velocity = MARKER_YAW_LR_SIGN * (
            lr_speed if marker_yaw > 0 else -lr_speed
        )

    # 上下控制
    if ey > xy_tolerance:
        up_down_velocity = -12 if ey < 90 else -18
    elif ey < -xy_tolerance:
        up_down_velocity = 12 if ey > -90 else 18

    # 前後距離控制
    dist_error = target_area - state["filtered_area"]
    abs_dist_error = abs(dist_error)

    if abs_dist_error <= dist_hold_tolerance:
        state["fb_hold"] = True
    elif abs_dist_error >= dist_release_tolerance:
        state["fb_hold"] = False

    if not state["fb_hold"]:
        fb_speed = 12 if abs_dist_error < 2800 else 20
        forward_backward_velocity = fb_speed if dist_error > 0 else -fb_speed

    # 限制 RC 更新頻率
    now = time.monotonic()
    if now - state["last_cmd_ts"] >= 0.08:
        state["last_cmd"] = (
            left_right_velocity,
            forward_backward_velocity,
            up_down_velocity,
            yaw_velocity,
        )
        state["last_cmd_ts"] = now

    tello.send_rc_control(*state["last_cmd"])

    yaw_ok = marker_yaw is not None and abs(marker_yaw) <= ANGLE_LOCK_TOLERANCE_DEG

    # 鎖定條件: 中心對準 + 距離進入 hold + yaw 正對
    is_locked = (
        abs(state["filtered_ex"]) <= xy_tolerance
        and abs(state["filtered_ey"]) <= xy_tolerance
        and state["fb_hold"]
        and yaw_ok
    )

    return int(state["filtered_area"]), int(dist_error), yaw_ok, is_locked


def track_marker_with_pose(
    tello,
    error_x,
    error_y,
    pose_x_cm,
    pose_y_cm,
    pose_z_cm,
    marker_yaw,
    state,
):
    state["filtered_x_cm"] = _low_pass(state["filtered_x_cm"], pose_x_cm)
    state["filtered_y_cm"] = _low_pass(state["filtered_y_cm"], pose_y_cm)
    state["filtered_z_cm"] = _low_pass(state["filtered_z_cm"], pose_z_cm)
    state["filtered_yaw_deg"] = _low_pass(state["filtered_yaw_deg"], marker_yaw, alpha=0.2)

    x_cm = state["filtered_x_cm"]
    y_cm = state["filtered_y_cm"]
    z_cm = state["filtered_z_cm"]
    yaw_deg = state["filtered_yaw_deg"]

    has_pose = (
        x_cm is not None
        and y_cm is not None
        and z_cm is not None
        and yaw_deg is not None
    )
    if not has_pose:
        tello.send_rc_control(0, 0, 0, 0)
        return None, None, None, None, False

    now = time.monotonic()
    last_pid_ts = state["last_pid_ts"]
    dt = 0.08 if last_pid_ts is None else now - last_pid_ts
    state["last_pid_ts"] = now

    x_error = x_cm - TARGET_X_CM
    y_error = y_cm - TARGET_Y_CM
    z_error = z_cm - TARGET_Z_CM
    yaw_error = yaw_deg

    pids = state["pids"]
    lr_velocity = 0
    fb_velocity = 0
    ud_velocity = 0
    yaw_velocity = 0

    vertical_locked = abs(error_y) <= XY_LOCK_TOLERANCE
    if not vertical_locked:
        if error_y > XY_LOCK_TOLERANCE:
            ud_velocity = -10 if error_y < 90 else -16
        elif error_y < -XY_LOCK_TOLERANCE:
            ud_velocity = 10 if error_y > -90 else 16

        for pid in pids.values():
            pid.reset()

        if now - state["last_cmd_ts"] >= 0.08:
            state["last_cmd"] = (0, 0, ud_velocity, 0)
            state["last_cmd_ts"] = now

        tello.send_rc_control(*state["last_cmd"])
        pose_errors = (x_error, y_error, z_error, yaw_error)
        return x_cm, y_cm, z_cm, pose_errors, False

    if abs(error_x) >= X_EDGE_GUARD:
        yaw_velocity = 8 if error_x > 0 else -8
        for pid in pids.values():
            pid.reset()

        if now - state["last_cmd_ts"] >= 0.08:
            state["last_cmd"] = (0, 0, ud_velocity, yaw_velocity)
            state["last_cmd_ts"] = now

        tello.send_rc_control(*state["last_cmd"])
        pose_errors = (x_error, y_error, z_error, yaw_error)
        return x_cm, y_cm, z_cm, pose_errors, False

    if abs(x_error) > POSE_X_TOLERANCE_CM:
        lr_velocity = POSE_LR_SIGN * _rc_speed(pids["x"].update(x_error, dt))
    else:
        pids["x"].reset()

    if abs(z_error) > POSE_Z_TOLERANCE_CM:
        fb_velocity = POSE_FB_SIGN * _rc_speed(pids["z"].update(z_error, dt))
    else:
        pids["z"].reset()

    if abs(y_error) > POSE_Y_TOLERANCE_CM:
        ud_velocity = POSE_UD_SIGN * _rc_speed(pids["y"].update(y_error, dt))
    else:
        pids["y"].reset()

    if abs(yaw_error) > POSE_YAW_TOLERANCE_DEG:
        yaw_velocity = POSE_YAW_SIGN * _rc_speed(pids["yaw"].update(yaw_error, dt))
    else:
        pids["yaw"].reset()

    if now - state["last_cmd_ts"] >= 0.08:
        state["last_cmd"] = (
            lr_velocity,
            fb_velocity,
            ud_velocity,
            yaw_velocity,
        )
        state["last_cmd_ts"] = now

    tello.send_rc_control(*state["last_cmd"])

    is_locked = (
        abs(x_error) <= POSE_X_TOLERANCE_CM
        and abs(y_error) <= POSE_Y_TOLERANCE_CM
        and abs(z_error) <= POSE_Z_TOLERANCE_CM
        and abs(yaw_error) <= POSE_YAW_TOLERANCE_DEG
    )

    pose_errors = (x_error, y_error, z_error, yaw_error)
    return x_cm, y_cm, z_cm, pose_errors, is_locked


def pulse_rc(tello, lr, fb, ud, yaw, duration=0.16):
    # 手動控制短脈衝
    tello.send_rc_control(lr, fb, ud, yaw)
    time.sleep(duration)
    tello.send_rc_control(0, 0, 0, 0)


def should_exit():
    key = cv2.waitKey(1) & 0xFF
    return key == ord('q') or key == 27


def safe_shutdown_tello(tello, is_flying):
    # djitellopy 的 __del__ 會再呼叫 end()，所以關閉後要同步狀態避免重複 streamoff。
    try:
        tello.send_rc_control(0, 0, 0, 0)
    except Exception:
        pass

    try:
        if is_flying:
            tello.land()
    except Exception:
        pass

    try:
        tello.stream_on = False
        tello.is_flying = False
        tello.end()
    except Exception:
        pass


def main() -> None:
    os.makedirs(PHOTO_DIR, exist_ok=True)
    # 靜音 djitellopy 的 INFO 日誌（例如 rc 指令輸出）
    for logger_name in ("djitellopy", "djitellopy.tello"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
        logger.disabled = True

    tello = Tello()
    is_flying = False
    auto_track = False
    target_takeoff_up_cm = 70
    current_target_index = 0
    inspection_done = False
    moving_to_next_face = False
    transition_mode = "SCAN"

    tracker_state = {
        "filtered_area": None,
        "filtered_ex": 0.0,
        "filtered_ey": 0.0,
        "filtered_x_cm": None,
        "filtered_y_cm": None,
        "filtered_z_cm": None,
        "filtered_yaw_deg": None,
        "fb_hold": True,
        "last_cmd": (0, 0, 0, 0),
        "last_cmd_ts": 0.0,
        "last_pid_ts": None,
        "lock_start_ts": None,
        "photo_taken": False,
        "pids": {
            "x": PIDController(kp=0.35, kd=0.02),
            "y": PIDController(kp=0.45, kd=0.02),
            "z": PIDController(kp=0.35, kd=0.02),
            "yaw": PIDController(kp=0.7, kd=0.03),
        },
    }

    try:
        Tello.FRAME_GRAB_TIMEOUT = 15
        tello.connect(wait_for_state=False)
        battery = tello.query_battery()
        print(f"Tello battery: {battery}%")

        try:
            tello.streamoff()
            time.sleep(0.5)
        except Exception:
            pass

        tello.streamon()
        time.sleep(2.0)
        frame_reader = tello.get_frame_read()

        print("Controls: t=takeoff, m=toggle auto/manual, l=land, q=quit")
        print("Manual: w/s=forward/back, a/d=left/right, r/f=up/down, z/c=yaw")
        print(f"Inspection marker IDs: {TARGET_MARKER_IDS}")

        while True:
            safe_target_index = min(current_target_index, len(TARGET_MARKER_IDS) - 1)
            current_target_id = TARGET_MARKER_IDS[safe_target_index]
            raw_frame = frame_reader.frame
            if raw_frame is None:
                if should_exit():
                    print("Exit key detected, shutting down...")
                    break
                time.sleep(0.02)
                continue

            # 修正顏色通道：避免黃變藍 (RGB -> BGR)
            if FRAME_IS_RGB:
                raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

            # clean_frame: 儲存用乾淨高畫質影像 (無任何疊字)
            clean_frame = raw_frame.copy()

            # display_frame: 顯示/偵測用
            frame = cv2.resize(raw_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            (
                error_x,
                error_y,
                marker_id,
                marker_area,
                marker_yaw,
                marker_pitch,
                pose_x_cm,
                pose_y_cm,
                pose_z_cm,
            ) = _detect_aruco_detail(frame, current_target_id)

            # 畫中心十字
            h, w = frame.shape[:2]
            frame_cx = w // 2
            frame_cy = h // 2
            cv2.drawMarker(
                frame,
                (frame_cx, frame_cy),
                (255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
            )

            # 顯示辨識資訊 (只在 display frame)
            if marker_id is not None:
                yaw_text = f"{marker_yaw:.1f}" if marker_yaw is not None else "N/A"
                pitch_text = (
                    f"{marker_pitch:.1f}" if marker_pitch is not None else "N/A"
                )
                pose_text = (
                    f"x:{pose_x_cm:.1f} y:{pose_y_cm:.1f} z:{pose_z_cm:.1f} cm"
                    if pose_x_cm is not None
                    else "pose: N/A"
                )
                center_locked = (
                    abs(error_x) <= XY_LOCK_TOLERANCE
                    and abs(error_y) <= XY_LOCK_TOLERANCE
                )
                yaw_locked = (
                    marker_yaw is not None
                    and abs(marker_yaw) <= ANGLE_LOCK_TOLERANCE_DEG
                )
                angle_locked = center_locked and yaw_locked
                angle_lock_text = (
                    f"FRONT LOCK: {'YES' if angle_locked else 'NO'} "
                    f"(center + yaw <= {ANGLE_LOCK_TOLERANCE_DEG:.0f} deg)"
                )
                angle_lock_color = (0, 255, 0) if angle_locked else (0, 165, 255)
                cv2.putText(
                    frame,
                    f"ID: {marker_id}  ex: {error_x}  ey: {error_y}  area: {marker_area}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Yaw: {yaw_text} deg  Pitch dbg: {pitch_text} deg",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    pose_text,
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    angle_lock_text,
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    angle_lock_color,
                    2,
                )
            else:
                cv2.putText(
                    frame,
                    f"Target ID {current_target_id} not detected",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            mode_name = "AUTO" if auto_track else "MANUAL"
            face_text = "DONE" if inspection_done else f"{current_target_index + 1}/{len(TARGET_MARKER_IDS)}"
            search_text = transition_mode if moving_to_next_face else "SCAN"
            status_text = (
                f"Flying: {is_flying}  Mode: {mode_name}  "
                f"Face: {face_text}  "
                f"Target ID: {current_target_id}  Search: {search_text}"
            )
            cv2.putText(
                frame,
                status_text,
                (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

            if is_flying and auto_track and not inspection_done:
                if marker_id is not None:
                    moving_to_next_face = False
                    transition_mode = "SCAN"
                    if USE_POSE_CONTROL:
                        x_cm, y_cm, z_cm, pose_errors, is_locked = track_marker_with_pose(
                            tello,
                            error_x,
                            error_y,
                            pose_x_cm,
                            pose_y_cm,
                            pose_z_cm,
                            marker_yaw,
                            tracker_state,
                        )
                        if pose_errors is None:
                            control_text = "pose control: waiting for solvePnP"
                        else:
                            ex_cm, ey_cm, ez_cm, eyaw_deg = pose_errors
                            if abs(error_y) > XY_LOCK_TOLERANCE:
                                control_text = (
                                    f"VERTICAL ex:{error_x} ey:{error_y} "
                                    f"pose z:{ez_cm:.1f}"
                                )
                            elif abs(error_x) >= X_EDGE_GUARD:
                                control_text = (
                                    f"EDGE GUARD ex:{error_x} ey:{error_y} "
                                    f"pose z:{ez_cm:.1f}"
                                )
                            else:
                                control_text = (
                                    f"err x:{ex_cm:.1f} y:{ey_cm:.1f} "
                                    f"z:{ez_cm:.1f} yaw:{eyaw_deg:.1f}"
                                )
                    else:
                        filtered_area, dist_error, yaw_ok, is_locked = (
                            track_marker_with_distance_stable(
                                tello,
                                error_x,
                                error_y,
                                marker_area,
                                marker_yaw,
                                tracker_state,
                            )
                        )
                        control_text = (
                            f"area mode f_area:{filtered_area} "
                            f"dist_err:{dist_error} yaw_ok:{yaw_ok}"
                        )
                    cv2.putText(
                        frame,
                        control_text,
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                    )

                    # 鎖定懸停連續 3 秒後自動拍照 (僅拍一次，直到 marker 遺失或重啟)
                    if is_locked:
                        if tracker_state["lock_start_ts"] is None:
                            tracker_state["lock_start_ts"] = time.monotonic()
                        else:
                            hold_time = time.monotonic() - tracker_state["lock_start_ts"]
                            cv2.putText(
                                frame,
                                f"LOCKING {hold_time:.1f}/3.0s",
                                (10, 175),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2,
                            )
                            if not tracker_state["photo_taken"] and hold_time >= 3.0:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = (
                                    f"block_{current_target_index + 1}_"
                                    f"id_{current_target_id}_{timestamp}.jpg"
                                )
                                photo_path = os.path.join(PHOTO_DIR, filename)
                                cv2.imwrite(
                                    photo_path,
                                    clean_frame,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                                )
                                tracker_state["photo_taken"] = True
                                print(f"Auto photo saved: {photo_path}")
                                current_target_index += 1
                                if current_target_index >= len(TARGET_MARKER_IDS):
                                    inspection_done = True
                                    auto_track = False
                                    tello.send_rc_control(0, 0, 0, 0)
                                    print("Inspection complete. Auto mode disabled.")
                                else:
                                    next_target_id = TARGET_MARKER_IDS[current_target_index]
                                    reset_tracker_state(tracker_state)
                                    tracker_state["photo_taken"] = False
                                    moving_to_next_face = True
                                    transition_mode = TRANSITION_MODES[
                                        current_target_index - 1
                                    ]
                                    print(
                                        f"Moving to target ID: {next_target_id} "
                                        f"mode: {transition_mode}"
                                    )
                    else:
                        tracker_state["lock_start_ts"] = None
                else:
                    # 拍完一面後，持續右移加左轉繞塔找下一面；初始搜尋則原地掃描。
                    if moving_to_next_face:
                        if transition_mode == "DOWN":
                            tello.send_rc_control(0, 0, SAME_FACE_DOWN_SPEED, 0)
                        elif transition_mode == "UP":
                            tello.send_rc_control(0, 0, SAME_FACE_UP_SPEED, 0)
                        else:
                            tello.send_rc_control(
                                NEXT_FACE_RIGHT_SPEED,
                                0,
                                0,
                                NEXT_FACE_LEFT_YAW_SPEED,
                            )
                    else:
                        tello.send_rc_control(0, 0, 0, SEARCH_YAW_SPEED)
                    cv2.putText(
                        frame,
                        f"SEARCHING ID {current_target_id} {transition_mode}...",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 165, 255),
                        2,
                    )

                    # 遺失目標 marker 就重置鎖定狀態，但保留已拍照資訊。
                    reset_tracker_state(tracker_state)

            elif is_flying and auto_track and inspection_done:
                tello.send_rc_control(0, 0, 0, 0)

            elif is_flying and not auto_track:
                # 手動模式未按鍵時懸停
                tello.send_rc_control(0, 0, 0, 0)

            cv2.imshow("Tello Camera", frame)
            key = cv2.waitKey(1) & 0xFF

            # t: 起飛 + 預設進入自動模式
            if key == ord('t'):
                if not is_flying:
                    tello.takeoff()
                    tello.move_up(target_takeoff_up_cm)
                    is_flying = True
                    auto_track = True
                    current_target_index = 0
                    inspection_done = False
                    moving_to_next_face = False
                    transition_mode = "SCAN"
                    reset_tracker_state(tracker_state)
                    tracker_state["photo_taken"] = False
                    print("Takeoff complete. Auto mode enabled.")

            # m: 自動/手動切換
            elif key == ord('m'):
                if is_flying:
                    auto_track = not auto_track
                    if not auto_track:
                        moving_to_next_face = False
                        transition_mode = "SCAN"
                    tello.send_rc_control(0, 0, 0, 0)
                    print(f"Switched to {'AUTO' if auto_track else 'MANUAL'} mode.")

            # 手動操作（僅手動模式）
            elif is_flying and not auto_track and key == ord('w'):
                pulse_rc(tello, 0, 24, 0, 0)
            elif is_flying and not auto_track and key == ord('s'):
                pulse_rc(tello, 0, -24, 0, 0)
            elif is_flying and not auto_track and key == ord('a'):
                pulse_rc(tello, -24, 0, 0, 0)
            elif is_flying and not auto_track and key == ord('d'):
                pulse_rc(tello, 24, 0, 0, 0)
            elif is_flying and not auto_track and key == ord('r'):
                pulse_rc(tello, 0, 0, 24, 0)
            elif is_flying and not auto_track and key == ord('f'):
                pulse_rc(tello, 0, 0, -24, 0)
            elif is_flying and not auto_track and key == ord('z'):
                pulse_rc(tello, 0, 0, 0, -28)
            elif is_flying and not auto_track and key == ord('c'):
                pulse_rc(tello, 0, 0, 0, 28)

            # l: 降落
            elif key == ord('l'):
                auto_track = False
                moving_to_next_face = False
                transition_mode = "SCAN"
                tello.send_rc_control(0, 0, 0, 0)
                if is_flying:
                    tello.land()
                    is_flying = False
                tracker_state["photo_taken"] = False
                reset_tracker_state(tracker_state)
                print("Landing complete.")

            # q / Esc: 安全退出
            elif key == ord('q') or key == 27:
                print("Exit key detected, shutting down...")
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt detected, shutting down...")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # 不論正常結束或發生例外，都嘗試安全停止並釋放資源
        safe_shutdown_tello(tello, is_flying)
        cv2.destroyAllWindows()
        print("Resources released. Program ended.")

if __name__ == "__main__":
    main()
