from djitellopy import Tello
import cv2
import numpy as np
import time
import os
import logging
from datetime import datetime


PHOTO_DIR = "photos"
PHOTO_LOCK_SECONDS = 0.5
PHOTO_LOCK_LOSS_GRACE_SEC = 0.3
HOME_MARKER_ID = 0
# 若畫面顏色正常可改成 False；若黃變藍請維持 True
FRAME_IS_RGB = True
TARGET_MARKER_IDS = [0, 4, 5, 1, 2, 6, 12, 8, 9, 13, 14, 10, 16, 15, 7, 3]
TRANSITION_MODES = [
    "DOWN",
    "NEXT",
    "UP",
    "NEXT",
    "DOWN",
    "BRIDGE_ROTATE",
    "UP",
    "NEXT",
    "DOWN",
    "NEXT",
    "UP",
    "NEXT",
    "DOWN",
    "BRIDGE_RIGHT",
    "UP",
]
USE_POSE_CONTROL = True
SEARCH_YAW_SPEED = -20
REACQUIRE_MAX_AGE_SEC = 1.2
REACQUIRE_YAW_SPEED = 12
REACQUIRE_UP_DOWN_SPEED = 12
REACQUIRE_EDGE_MARGIN = 70

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
TOWER_FRONT_BACK_WIDTH_CM = 33.0
TOWER_LEFT_RIGHT_WIDTH_CM = 25.0
SECOND_TOWER_CENTER_Z_CM = 185.0
TOWER_UPPER_MARKER_Y_CM = 94.0
TOWER_LOWER_MARKER_Y_CM = 27.5
TOWER_DISPLAY_X_SIGN = 1.0
TOWER_WAYPOINT_TOLERANCE_CM = 12.0
TOWER_VERTICAL_TOLERANCE_CM = 7.0
TOWER_HEADING_TOLERANCE_DEG = 12.0
TOWER_ARC_TOLERANCE_CM = 16.0
TOWER_ARC_VERTICAL_TOLERANCE_CM = 10.0
TOWER_ARC_HEADING_TOLERANCE_DEG = 18.0
TOWER_WAYPOINT_SOFT_TIMEOUT_SEC = 6.0
TOWER_WAYPOINT_SOFT_DISTANCE_CM = 28.0
TOWER_WAYPOINT_SOFT_HEADING_DEG = 30.0
TOWER_WAYPOINT_YAW_SCALE = 1.0
TOWER_WAYPOINT_MAX_YAW_SPEED = 24
TOWER_WAYPOINT_UD_BOOST = 1.5
TOWER_YAW_SIGN = -1  # 塔座標角度逆時針為正；Tello RC yaw 的正方向相反。
TOWER_UD_SIGN = 1  # 塔座標 Y 向上為正，與 Tello RC up/down 相同。
TARGET_ACQUIRE_YAW_SPEED = 20
TARGET_ACQUIRE_VERTICAL_SPEED = 8
BRIDGE_ROTATE_YAW_SPEED = 32
BRIDGE_RIGHT_ACQUIRE_SPEED = 8
NAVIGATION_BLIND_MAX_SEC = 0.6
DEAD_RECKONING_MAX_CONTROL_SEC = 6.0
DEAD_RECKONING_MAX_STEP_SEC = 0.5
RC_LR_SPEED_SCALE = 1.0
RC_FB_SPEED_SCALE = 1.0
RC_UD_SPEED_SCALE = 1.0
RC_YAW_SPEED_SCALE = 1.0
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
WAYPOINT_YAW_SLOWDOWN_DEG = 18.0
WAYPOINT_YAW_PRIORITY_DEG = 35.0
WAYPOINT_YAW_SLOWDOWN_SCALE = 0.45
WAYPOINT_YAW_PRIORITY_SCALE = 0.25

CAMERA_MATRIX = np.array(
    [
        [FOCAL_LENGTH_PX, 0.0, DISPLAY_WIDTH / 2.0],
        [0.0, FOCAL_LENGTH_PX, DISPLAY_HEIGHT / 2.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
DIST_COEFFS = np.zeros((5, 1), dtype=np.float32)


def _build_tower_marker_map():
    half_front_width = TOWER_FRONT_BACK_WIDTH_CM / 2.0
    half_side_width = TOWER_LEFT_RIGHT_WIDTH_CM / 2.0
    local_face_defs = {
        1: {
            "center_x": 0.0,
            "center_z": -half_side_width,
            "normal_deg": -90.0,
            "upper_offset": 0,
            "lower_offset": 4,
            "width_cm": TOWER_FRONT_BACK_WIDTH_CM,
        },
        2: {
            "center_x": half_front_width,
            "center_z": 0.0,
            "normal_deg": 0.0,
            "upper_offset": 1,
            "lower_offset": 5,
            "width_cm": TOWER_LEFT_RIGHT_WIDTH_CM,
        },
        3: {
            "center_x": 0.0,
            "center_z": half_side_width,
            "normal_deg": 90.0,
            "upper_offset": 2,
            "lower_offset": 6,
            "width_cm": TOWER_FRONT_BACK_WIDTH_CM,
        },
        4: {
            "center_x": -half_front_width,
            "center_z": 0.0,
            "normal_deg": 180.0,
            "upper_offset": 3,
            "lower_offset": 7,
            "width_cm": TOWER_LEFT_RIGHT_WIDTH_CM,
        },
    }
    tower_defs = (
        {"tower_id": 1, "id_offset": 0, "center_x": 0.0, "center_z": 0.0},
        {
            "tower_id": 2,
            "id_offset": 8,
            "center_x": 0.0,
            "center_z": SECOND_TOWER_CENTER_Z_CM,
        },
    )

    marker_map = {}
    for tower in tower_defs:
        for face_index, face in local_face_defs.items():
            for row_name, marker_offset, vertical_cm in (
                ("UP", face["upper_offset"], TOWER_UPPER_MARKER_Y_CM),
                ("DOWN", face["lower_offset"], TOWER_LOWER_MARKER_Y_CM),
            ):
                marker_id = tower["id_offset"] + marker_offset
                marker_map[marker_id] = {
                    "tower_id": tower["tower_id"],
                    "tower_center_x_cm": tower["center_x"],
                    "tower_center_z_cm": tower["center_z"],
                    "face": face_index,
                    "row": row_name,
                    "x_cm": tower["center_x"] + face["center_x"],
                    "z_cm": tower["center_z"] + face["center_z"],
                    "vertical_cm": vertical_cm,
                    "normal_deg": face["normal_deg"],
                    "width_cm": face["width_cm"],
                }

    # 第二座塔第 4 面上排的 ID 11 已更換為 ID 16。
    marker_map[16] = marker_map.pop(11)
    return marker_map


TOWER_MARKER_MAP = _build_tower_marker_map()
BRIDGE_LOCALIZATION_MARKER_IDS = {
    "BRIDGE_ROTATE": {2, 6, 8, 12},
    "BRIDGE_RIGHT": {3, 7, 15, 16},
}


def marker_allowed_for_localization(
    marker_id,
    current_target_id,
    moving_to_next_face,
    transition_mode,
):
    if marker_id not in TOWER_MARKER_MAP:
        return False
    if not moving_to_next_face:
        return True

    bridge_marker_ids = BRIDGE_LOCALIZATION_MARKER_IDS.get(transition_mode)
    if bridge_marker_ids is not None:
        return marker_id in bridge_marker_ids

    marker_info = TOWER_MARKER_MAP[marker_id]
    target_info = TOWER_MARKER_MAP.get(current_target_id)
    return (
        target_info is not None
        and marker_info["tower_id"] == target_info["tower_id"]
    )


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


def _rc_speed(value, min_speed=MIN_AUTO_SPEED, max_speed=MAX_AUTO_SPEED):
    speed = int(round(value))
    if speed == 0:
        return 0

    if abs(speed) < min_speed:
        speed = min_speed if speed > 0 else -min_speed

    return int(np.clip(speed, -max_speed, max_speed))


def _low_pass(previous, current, alpha=0.25):
    if current is None:
        return previous
    if previous is None:
        return float(current)
    return (1.0 - alpha) * previous + alpha * float(current)


def _marker_error_to_body_error(marker_x_error, marker_z_error, yaw_deg):
    yaw_rad = np.radians(yaw_deg)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)

    body_right_error = cos_yaw * marker_x_error + sin_yaw * marker_z_error
    body_forward_error = -sin_yaw * marker_x_error + cos_yaw * marker_z_error
    return body_right_error, body_forward_error


def _unit_vector_from_degrees(angle_deg):
    angle_rad = np.radians(angle_deg)
    return np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32)


def _marker_to_tower_rotation(marker_info):
    normal_vec = _unit_vector_from_degrees(marker_info["normal_deg"])
    right_vec = _unit_vector_from_degrees(marker_info["normal_deg"] + 90.0)
    return np.array(
        [
            [right_vec[0], 0.0, normal_vec[0]],
            [0.0, 1.0, 0.0],
            [right_vec[1], 0.0, normal_vec[1]],
        ],
        dtype=np.float64,
    )


def estimate_tower_pose(marker_id, rvec, tvec):
    marker_info = TOWER_MARKER_MAP.get(marker_id)
    if marker_info is None or rvec is None or tvec is None:
        return None

    marker_to_camera, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    marker_in_camera_cm = (
        np.asarray(tvec, dtype=np.float64).reshape(3) * 100.0
    )
    camera_in_marker_cm = (
        -marker_to_camera.T @ np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    ).reshape(3) * 100.0

    marker_to_tower = _marker_to_tower_rotation(marker_info)
    marker_pos = np.array(
        [
            marker_info["x_cm"],
            marker_info["vertical_cm"],
            marker_info["z_cm"],
        ],
        dtype=np.float64,
    )
    drone_pos = marker_pos + marker_to_tower @ camera_in_marker_cm
    tower_y_full_cm = float(drone_pos[1])

    # 平面 marker 的 PnP pitch 容易把前後距離誤差混入完整矩陣 Y。
    # 塔座標 Y 改用 marker 絕對高度加相機座標中的垂直位移。
    marker_relative_y_cm = float(marker_in_camera_cm[1])
    tower_y_cm = float(marker_info["vertical_cm"] + marker_relative_y_cm)

    # R.T 將相機座標轉回 marker 座標，再轉進塔座標。
    camera_to_tower = marker_to_tower @ marker_to_camera.T
    camera_forward_tower = camera_to_tower @ np.array([0.0, 0.0, 1.0])
    horizontal_forward = np.hypot(
        camera_forward_tower[0],
        camera_forward_tower[2],
    )
    if horizontal_forward < 1e-6:
        return None

    heading_deg = _normalize_angle_degrees(
        np.degrees(
            np.arctan2(camera_forward_tower[2], camera_forward_tower[0])
        )
    )
    normal_vec = _unit_vector_from_degrees(marker_info["normal_deg"])
    target_pos = np.array([marker_info["x_cm"], marker_info["z_cm"]])
    target_pos = target_pos + normal_vec * TARGET_Z_CM
    target_error = target_pos - drone_pos[[0, 2]]
    target_y_cm = marker_info["vertical_cm"]

    return {
        "tower_x_cm": float(drone_pos[0]),
        "tower_y_cm": tower_y_cm,
        "tower_y_full_cm": tower_y_full_cm,
        "marker_relative_y_cm": marker_relative_y_cm,
        "tower_z_cm": float(drone_pos[2]),
        "display_tower_x_cm": float(drone_pos[0] * TOWER_DISPLAY_X_SIGN),
        "heading_deg": heading_deg,
        "target_x_cm": float(target_pos[0]),
        "target_y_cm": float(target_y_cm),
        "target_z_cm": float(target_pos[1]),
        "target_error_x_cm": float(target_error[0]),
        "target_error_y_cm": float(target_y_cm - tower_y_cm),
        "target_error_z_cm": float(target_error[1]),
        "face": marker_info["face"],
        "row": marker_info["row"],
    }


def tower_marker_target_position(marker_id):
    marker_info = TOWER_MARKER_MAP.get(marker_id)
    if marker_info is None:
        return None

    normal_vec = _unit_vector_from_degrees(marker_info["normal_deg"])
    marker_pos = np.array([marker_info["x_cm"], marker_info["z_cm"]], dtype=np.float32)
    target_pos = marker_pos + normal_vec * TARGET_Z_CM
    return {
        "x_cm": float(target_pos[0]),
        "y_cm": float(marker_info["vertical_cm"]),
        "z_cm": float(target_pos[1]),
        "heading_deg": _normalize_angle_degrees(marker_info["normal_deg"] + 180.0),
        "face": marker_info["face"],
        "row": marker_info["row"],
    }


def build_tower_transition_waypoints(from_marker_id, to_marker_id):
    from_info = TOWER_MARKER_MAP.get(from_marker_id)
    to_info = TOWER_MARKER_MAP.get(to_marker_id)
    target = tower_marker_target_position(to_marker_id)
    if from_info is None or to_info is None or target is None:
        return []

    if (from_marker_id, to_marker_id) == (6, 12):
        return [
            dict(
                target,
                label="BRIDGE TURN",
                forced_yaw_sign=-1,
                forced_yaw_speed=BRIDGE_ROTATE_YAW_SPEED,
                position_tolerance_cm=TOWER_ARC_TOLERANCE_CM,
                vertical_tolerance_cm=TOWER_ARC_VERTICAL_TOLERANCE_CM,
                heading_tolerance_deg=TOWER_ARC_HEADING_TOLERANCE_DEG,
            )
        ]

    if (from_marker_id, to_marker_id) == (15, 7):
        return [
            dict(
                target,
                label="BRIDGE RIGHT",
                position_tolerance_cm=TOWER_ARC_TOLERANCE_CM,
                vertical_tolerance_cm=TOWER_ARC_VERTICAL_TOLERANCE_CM,
                heading_tolerance_deg=TOWER_ARC_HEADING_TOLERANCE_DEG,
            )
        ]

    same_tower = from_info["tower_id"] == to_info["tower_id"]
    if same_tower and from_info["face"] == to_info["face"]:
        return [dict(target, label="SAME")]

    if not same_tower:
        return [dict(target, label="DIRECT")]

    next_face = from_info["face"] % 4 + 1
    if to_info["face"] != next_face:
        return [dict(target, label="DIRECT")]

    half_front = TOWER_FRONT_BACK_WIDTH_CM / 2.0
    half_side = TOWER_LEFT_RIGHT_WIDTH_CM / 2.0
    corner_by_face = {
        1: (half_front, -half_side),
        2: (half_front, half_side),
        3: (-half_front, half_side),
        4: (-half_front, -half_side),
    }
    start_angle = from_info["normal_deg"]
    end_angle = start_angle + 90.0
    start_y = from_info["vertical_cm"]
    end_y = to_info["vertical_cm"]
    local_corner_x, local_corner_z = corner_by_face[from_info["face"]]
    corner_x = from_info["tower_center_x_cm"] + local_corner_x
    corner_z = from_info["tower_center_z_cm"] + local_corner_z

    waypoints = []
    for index, fraction in enumerate((0.25, 0.75)):
        angle = start_angle + (end_angle - start_angle) * fraction
        offset = _unit_vector_from_degrees(angle) * TARGET_Z_CM
        waypoints.append(
            {
                "x_cm": float(corner_x + offset[0]),
                "y_cm": float(start_y + (end_y - start_y) * fraction),
                "z_cm": float(corner_z + offset[1]),
                "heading_deg": _normalize_angle_degrees(angle + 180.0),
                "face": to_info["face"],
                "row": to_info["row"],
                "label": f"ARC{index + 1}",
                "position_tolerance_cm": TOWER_ARC_TOLERANCE_CM,
                "vertical_tolerance_cm": TOWER_ARC_VERTICAL_TOLERANCE_CM,
                "heading_tolerance_deg": TOWER_ARC_HEADING_TOLERANCE_DEG,
            }
        )

    waypoints.append(
        dict(
            target,
            label="FACE",
            position_tolerance_cm=TOWER_ARC_TOLERANCE_CM,
            vertical_tolerance_cm=TOWER_ARC_VERTICAL_TOLERANCE_CM,
            heading_tolerance_deg=TOWER_ARC_HEADING_TOLERANCE_DEG,
        )
    )
    return waypoints


def tower_error_to_body_error(tower_error_x, tower_error_z, heading_deg):
    heading_rad = np.radians(heading_deg)
    right_vec = np.array([np.sin(heading_rad), -np.cos(heading_rad)], dtype=np.float32)
    forward_vec = np.array([np.cos(heading_rad), np.sin(heading_rad)], dtype=np.float32)
    tower_error = np.array([tower_error_x, tower_error_z], dtype=np.float32)
    return float(np.dot(tower_error, right_vec)), float(np.dot(tower_error, forward_vec))


def _normalize_angle_degrees(angle):
    return float((angle + 180.0) % 360.0 - 180.0)


def predict_tower_pose(estimator, rc_command, now=None):
    now = time.monotonic() if now is None else now
    last_update_ts = estimator.get("last_update_ts")
    estimator["last_update_ts"] = now
    if last_update_ts is None or not estimator.get("initialized", False):
        return

    dt = float(np.clip(now - last_update_ts, 0.0, DEAD_RECKONING_MAX_STEP_SEC))
    if dt <= 0.0:
        return

    lr, fb, ud, yaw = rc_command
    right_speed = lr * RC_LR_SPEED_SCALE
    forward_speed = fb * RC_FB_SPEED_SCALE
    up_speed = ud * RC_UD_SPEED_SCALE
    heading_rad = np.radians(estimator["heading_deg"])

    estimator["tower_x_cm"] += (
        right_speed * np.sin(heading_rad)
        + forward_speed * np.cos(heading_rad)
    ) * dt
    estimator["tower_y_cm"] += up_speed * dt
    estimator["tower_z_cm"] += (
        -right_speed * np.cos(heading_rad)
        + forward_speed * np.sin(heading_rad)
    ) * dt
    estimator["heading_deg"] = _normalize_angle_degrees(
        estimator["heading_deg"] - yaw * RC_YAW_SPEED_SCALE * dt
    )


def correct_tower_pose_estimate(estimator, tower_pose, marker_id, now=None):
    if tower_pose is None:
        return

    now = time.monotonic() if now is None else now
    estimator["initialized"] = True
    estimator["tower_x_cm"] = tower_pose["tower_x_cm"]
    estimator["tower_y_cm"] = tower_pose["tower_y_cm"]
    estimator["tower_z_cm"] = tower_pose["tower_z_cm"]
    estimator["heading_deg"] = tower_pose["heading_deg"]
    estimator["last_marker_ts"] = now
    estimator["last_marker_id"] = marker_id
    estimator["last_update_ts"] = now


def get_estimated_tower_pose(estimator):
    if not estimator.get("initialized", False):
        return None

    return {
        "tower_x_cm": float(estimator["tower_x_cm"]),
        "tower_y_cm": float(estimator["tower_y_cm"]),
        "tower_z_cm": float(estimator["tower_z_cm"]),
        "display_tower_x_cm": float(
            estimator["tower_x_cm"] * TOWER_DISPLAY_X_SIGN
        ),
        "heading_deg": float(estimator["heading_deg"]),
    }


def dead_reckoning_age(estimator, now=None):
    last_marker_ts = estimator.get("last_marker_ts")
    if last_marker_ts is None:
        return float("inf")
    now = time.monotonic() if now is None else now
    return max(0.0, now - last_marker_ts)


def remember_rc_command(state, command, estimation_command=None):
    state["last_cmd"] = tuple(int(value) for value in command)
    if estimation_command is None:
        estimation_command = command
    state["last_estimation_cmd"] = tuple(
        int(value) for value in estimation_command
    )
    state["last_cmd_ts"] = time.monotonic()


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
        return (None,) * 11

    # 在顯示畫面上畫出 marker 框線
    cv2.aruco.drawDetectedMarkers(img, corners, ids)

    # 多個 marker 同時出現時，優先挑目前要拍的 target_id。
    marker_index = 0
    if target_id is not None:
        flat_ids = ids.flatten()
        matches = np.where(flat_ids == target_id)[0]
        if len(matches) == 0:
            return (None,) * 11
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
    pose_rvec = None
    pose_tvec = None
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
            pose_rvec = rvec.copy()
            pose_tvec = tvec.copy()

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
        pose_rvec = None
        pose_tvec = None

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
        pose_rvec,
        pose_tvec,
    )


def find_aruco(img):
    # 依需求提供的基礎函式：只回傳 error_x, error_y, marker_id
    # 內部仍沿用同一套偵測流程與畫框邏輯
    error_x, error_y, marker_id, _, _, _, _, _, _, _, _ = _detect_aruco_detail(img)
    if marker_id is None:
        return None, None, None
    return error_x, error_y, marker_id


def reset_tracker_state(state, preserve_command=False):
    previous_command = state.get("last_cmd", (0, 0, 0, 0))
    previous_estimation_command = state.get(
        "last_estimation_cmd",
        previous_command,
    )
    previous_command_ts = state.get("last_cmd_ts", 0.0)
    state["filtered_area"] = None
    state["filtered_ex"] = 0.0
    state["filtered_ey"] = 0.0
    state["filtered_x_cm"] = None
    state["filtered_y_cm"] = None
    state["filtered_z_cm"] = None
    state["filtered_yaw_deg"] = None
    state["fb_hold"] = True
    state["last_cmd"] = (0, 0, 0, 0)
    state["last_estimation_cmd"] = (0, 0, 0, 0)
    state["lock_accumulated_sec"] = 0.0
    state["lock_last_sample_ts"] = None
    state["lock_last_valid_ts"] = None
    state["last_pid_ts"] = None

    if preserve_command:
        state["last_cmd"] = previous_command
        state["last_estimation_cmd"] = previous_estimation_command
        state["last_cmd_ts"] = previous_command_ts

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
        remember_rc_command(
            state,
            (
                left_right_velocity,
                forward_backward_velocity,
                up_down_velocity,
                yaw_velocity,
            ),
        )

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
        remember_rc_command(state, (0, 0, 0, 0))
        tello.send_rc_control(0, 0, 0, 0)
        return None, None, None, None, False

    now = time.monotonic()
    last_pid_ts = state["last_pid_ts"]
    dt = 0.08 if last_pid_ts is None else now - last_pid_ts
    state["last_pid_ts"] = now

    marker_x_error = x_cm - TARGET_X_CM
    y_error = y_cm - TARGET_Y_CM
    marker_z_error = z_cm - TARGET_Z_CM
    yaw_error = yaw_deg
    body_right_error, body_forward_error = _marker_error_to_body_error(
        marker_x_error,
        marker_z_error,
        yaw_error,
    )

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
            remember_rc_command(state, (0, 0, ud_velocity, 0))

        tello.send_rc_control(*state["last_cmd"])
        pose_errors = (body_right_error, y_error, body_forward_error, yaw_error)
        return x_cm, y_cm, z_cm, pose_errors, False

    if abs(error_x) >= X_EDGE_GUARD:
        yaw_velocity = 8 if error_x > 0 else -8
        for pid in pids.values():
            pid.reset()

        if now - state["last_cmd_ts"] >= 0.08:
            remember_rc_command(state, (0, 0, ud_velocity, yaw_velocity))

        tello.send_rc_control(*state["last_cmd"])
        pose_errors = (body_right_error, y_error, body_forward_error, yaw_error)
        return x_cm, y_cm, z_cm, pose_errors, False

    yaw_scale = 1.0
    if abs(yaw_error) > WAYPOINT_YAW_PRIORITY_DEG:
        yaw_scale = WAYPOINT_YAW_PRIORITY_SCALE
    elif abs(yaw_error) > WAYPOINT_YAW_SLOWDOWN_DEG:
        yaw_scale = WAYPOINT_YAW_SLOWDOWN_SCALE

    if abs(body_right_error) > POSE_X_TOLERANCE_CM:
        lr_velocity = POSE_LR_SIGN * _rc_speed(
            pids["x"].update(body_right_error, dt)
        )
    else:
        pids["x"].reset()

    if abs(body_forward_error) > POSE_Z_TOLERANCE_CM:
        fb_velocity = POSE_FB_SIGN * _rc_speed(
            pids["z"].update(body_forward_error, dt)
        )
    else:
        pids["z"].reset()

    if abs(y_error) > POSE_Y_TOLERANCE_CM:
        ud_velocity = POSE_UD_SIGN * _rc_speed(pids["y"].update(y_error, dt))
    else:
        pids["y"].reset()

    if abs(yaw_error) > POSE_YAW_TOLERANCE_DEG:
        yaw_velocity = POSE_YAW_SIGN * _rc_speed(
            pids["yaw"].update(yaw_error, dt) * yaw_scale
        )
    else:
        pids["yaw"].reset()

    if now - state["last_cmd_ts"] >= 0.08:
        remember_rc_command(
            state,
            (
                lr_velocity,
                fb_velocity,
                ud_velocity,
                yaw_velocity,
            ),
        )

    tello.send_rc_control(*state["last_cmd"])

    is_locked = (
        abs(marker_x_error) <= POSE_X_TOLERANCE_CM
        and abs(y_error) <= POSE_Y_TOLERANCE_CM
        and abs(marker_z_error) <= POSE_Z_TOLERANCE_CM
        and abs(yaw_error) <= POSE_YAW_TOLERANCE_DEG
    )

    pose_errors = (body_right_error, y_error, body_forward_error, yaw_error)
    return x_cm, y_cm, z_cm, pose_errors, is_locked


def fly_tower_waypoint(tello, tower_pose, target_position, state):
    if tower_pose is None or target_position is None:
        return None, None, False

    error_x = target_position["x_cm"] - tower_pose["tower_x_cm"]
    error_y = target_position["y_cm"] - tower_pose["tower_y_cm"]
    error_z = target_position["z_cm"] - tower_pose["tower_z_cm"]
    heading_error = _normalize_angle_degrees(
        target_position["heading_deg"] - tower_pose["heading_deg"]
    )
    position_tolerance = target_position.get(
        "position_tolerance_cm",
        TOWER_WAYPOINT_TOLERANCE_CM,
    )
    vertical_tolerance = target_position.get(
        "vertical_tolerance_cm",
        TOWER_VERTICAL_TOLERANCE_CM,
    )
    heading_tolerance = target_position.get(
        "heading_tolerance_deg",
        TOWER_HEADING_TOLERANCE_DEG,
    )
    body_lr_error, body_fb_error = tower_error_to_body_error(
        error_x,
        error_z,
        tower_pose["heading_deg"],
    )

    now = time.monotonic()
    last_pid_ts = state["last_pid_ts"]
    dt = 0.08 if last_pid_ts is None else now - last_pid_ts
    state["last_pid_ts"] = now

    pids = state["pids"]
    waypoint_key = (
        round(target_position["x_cm"], 1),
        round(target_position["y_cm"], 1),
        round(target_position["z_cm"], 1),
        round(target_position["heading_deg"], 1),
    )
    if state.get("waypoint_key") != waypoint_key:
        state["waypoint_key"] = waypoint_key
        state["waypoint_start_ts"] = now
    waypoint_elapsed = now - state.get("waypoint_start_ts", now)

    lr_velocity = 0
    fb_velocity = 0
    ud_velocity = 0
    yaw_velocity = 0
    if abs(body_lr_error) > position_tolerance:
        lr_velocity = POSE_LR_SIGN * _rc_speed(pids["x"].update(body_lr_error, dt))
    else:
        pids["x"].reset()

    if abs(body_fb_error) > position_tolerance:
        fb_velocity = POSE_FB_SIGN * _rc_speed(pids["z"].update(body_fb_error, dt))
    else:
        pids["z"].reset()

    logical_ud_velocity = 0
    if abs(error_y) > vertical_tolerance:
        logical_ud_velocity = TOWER_UD_SIGN * _rc_speed(
            pids["y"].update(error_y, dt)
        )
        ud_velocity = int(
            np.clip(
                round(logical_ud_velocity * TOWER_WAYPOINT_UD_BOOST),
                -100,
                100,
            )
        )
    else:
        pids["y"].reset()

    position_reached = (
        abs(error_x) <= position_tolerance
        and abs(error_y) <= vertical_tolerance
        and abs(error_z) <= position_tolerance
    )
    if abs(heading_error) > heading_tolerance:
        yaw_output = _rc_speed(
            pids["waypoint_yaw"].update(heading_error, dt)
            * TOWER_WAYPOINT_YAW_SCALE,
            min_speed=6,
            max_speed=TOWER_WAYPOINT_MAX_YAW_SPEED,
        )
        forced_yaw_sign = target_position.get("forced_yaw_sign")
        if forced_yaw_sign is None:
            yaw_velocity = TOWER_YAW_SIGN * yaw_output
        else:
            forced_yaw_speed = target_position.get("forced_yaw_speed")
            if forced_yaw_speed is None:
                forced_yaw_speed = abs(yaw_output)
            yaw_velocity = int(np.sign(forced_yaw_sign) * forced_yaw_speed)
    else:
        pids["waypoint_yaw"].reset()

    if now - state["last_cmd_ts"] >= 0.08:
        command = (
            lr_velocity,
            fb_velocity,
            ud_velocity,
            yaw_velocity,
        )
        estimation_command = (
            lr_velocity,
            fb_velocity,
            logical_ud_velocity,
            yaw_velocity,
        )
        remember_rc_command(
            state,
            command,
            estimation_command=estimation_command,
        )

    tello.send_rc_control(*state["last_cmd"])
    reached = position_reached and abs(heading_error) <= heading_tolerance
    soft_reached = (
        waypoint_elapsed >= TOWER_WAYPOINT_SOFT_TIMEOUT_SEC
        and np.linalg.norm([error_x, error_y, error_z])
        <= TOWER_WAYPOINT_SOFT_DISTANCE_CM
        and abs(error_y) <= TOWER_ARC_VERTICAL_TOLERANCE_CM
        and abs(heading_error) <= TOWER_WAYPOINT_SOFT_HEADING_DEG
    )
    reached = reached or soft_reached
    errors = (
        error_x,
        error_y,
        error_z,
        body_lr_error,
        body_fb_error,
        heading_error,
    )
    return errors, state["last_cmd"], reached


def search_for_transition_target(tello, transition_mode, acquire_start_ts, state):
    elapsed = max(0.0, time.monotonic() - acquire_start_ts)
    sweep_phase = elapsed % 3.2
    sweep_sign = 1 if sweep_phase < 1.2 or sweep_phase >= 2.4 else -1

    if transition_mode == "BRIDGE_ROTATE":
        command = (0, 0, 0, -BRIDGE_ROTATE_YAW_SPEED)
    elif transition_mode == "BRIDGE_RIGHT":
        command = (BRIDGE_RIGHT_ACQUIRE_SPEED, 0, 0, 0)
    elif transition_mode in ("DOWN", "UP"):
        primary_sign = -1 if transition_mode == "DOWN" else 1
        up_down_velocity = (
            primary_sign * sweep_sign * TARGET_ACQUIRE_VERTICAL_SPEED
        )
        command = (0, 0, up_down_velocity, 0)
    else:
        # 一般 NEXT 固定左旋，避免轉回剛拍完的面。
        yaw_velocity = -TARGET_ACQUIRE_YAW_SPEED
        command = (0, 0, 0, yaw_velocity)

    remember_rc_command(state, command)
    tello.send_rc_control(*command)
    return command


def continue_navigation_blind(tello, state, transition_mode):
    last_seen_ts = state.get("last_seen_ts")
    age = float("inf") if last_seen_ts is None else time.monotonic() - last_seen_ts

    if age <= NAVIGATION_BLIND_MAX_SEC:
        command = tuple(int(np.clip(value, -10, 10)) for value in state["last_cmd"])
    else:
        # 失去三維定位後不再單向飛行，只在預期方向附近小幅搜尋 marker。
        search_start_ts = last_seen_ts if last_seen_ts is not None else time.monotonic()
        return search_for_transition_target(
            tello,
            transition_mode,
            search_start_ts,
            state,
        )

    remember_rc_command(state, command)
    tello.send_rc_control(*command)
    return command


def pulse_rc(tello, lr, fb, ud, yaw, state, estimator, duration=0.16):
    # 手動控制短脈衝
    command = (lr, fb, ud, yaw)
    predict_tower_pose(estimator, state["last_cmd"])
    remember_rc_command(state, command)
    tello.send_rc_control(*command)
    time.sleep(duration)
    predict_tower_pose(estimator, command)
    remember_rc_command(state, (0, 0, 0, 0))
    tello.send_rc_control(0, 0, 0, 0)


def should_exit():
    key = cv2.waitKey(1) & 0xFF
    return key == ord('q') or key == 27


def reacquire_lost_marker(tello, state, frame_width, frame_height):
    last_seen_ts = state.get("last_seen_ts")
    if last_seen_ts is None:
        return False, None

    if time.monotonic() - last_seen_ts > REACQUIRE_MAX_AGE_SEC:
        return False, None

    last_error_x = state.get("last_seen_error_x", 0)
    last_error_y = state.get("last_seen_error_y", 0)

    yaw_velocity = 0
    up_down_velocity = 0
    if last_error_x > frame_width / 2 - REACQUIRE_EDGE_MARGIN:
        yaw_velocity = REACQUIRE_YAW_SPEED
    elif last_error_x < -(frame_width / 2 - REACQUIRE_EDGE_MARGIN):
        yaw_velocity = -REACQUIRE_YAW_SPEED

    if last_error_y > frame_height / 2 - REACQUIRE_EDGE_MARGIN:
        up_down_velocity = -REACQUIRE_UP_DOWN_SPEED
    elif last_error_y < -(frame_height / 2 - REACQUIRE_EDGE_MARGIN):
        up_down_velocity = REACQUIRE_UP_DOWN_SPEED

    if yaw_velocity == 0 and up_down_velocity == 0:
        yaw_velocity = REACQUIRE_YAW_SPEED if last_error_x >= 0 else -REACQUIRE_YAW_SPEED

    command = (0, 0, up_down_velocity, yaw_velocity)
    remember_rc_command(state, command)
    tello.send_rc_control(*command)
    return True, command


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


def _draw_translucent_panel(frame, top, bottom, color=(18, 22, 28), alpha=0.78):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, top), (frame.shape[1], bottom), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, frame)


def _put_fitted_text(
    frame,
    text,
    origin,
    max_width,
    scale=0.5,
    color=(245, 245, 245),
    thickness=1,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fitted_scale = scale
    while fitted_scale > 0.32:
        width = cv2.getTextSize(text, font, fitted_scale, thickness)[0][0]
        if width <= max_width:
            break
        fitted_scale -= 0.03
    cv2.putText(
        frame,
        text,
        origin,
        font,
        fitted_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_flight_hud(
    frame,
    mode_name,
    progress_text,
    target_id,
    battery_percent,
    estimated_tower_pose,
    pose_source,
    pose_color,
    action_title,
    action_detail,
    action_color,
):
    width, height = frame.shape[1], frame.shape[0]
    _draw_translucent_panel(frame, 0, 61)
    _draw_translucent_panel(frame, height - 57, height)

    mode_color = (255, 210, 70) if mode_name == "AUTO" else (80, 190, 255)
    cv2.rectangle(frame, (0, 0), (4, 61), mode_color, -1)
    header = f"{mode_name}  {progress_text}  ID {target_id}"
    _put_fitted_text(frame, header, (10, 20), width - 105, 0.5, mode_color, 1)

    if battery_percent is None:
        battery_text = "--%"
        battery_level = 0.0
        battery_color = (150, 150, 150)
    else:
        battery_value = int(np.clip(battery_percent, 0, 100))
        battery_text = f"{battery_value}%"
        battery_level = battery_value / 100.0
        if battery_value <= 20:
            battery_color = (70, 70, 255)
        elif battery_value <= 40:
            battery_color = (40, 190, 255)
        else:
            battery_color = (90, 220, 120)

    cv2.putText(
        frame,
        battery_text,
        (width - 76, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        battery_color,
        1,
        cv2.LINE_AA,
    )
    battery_left = width - 37
    cv2.rectangle(frame, (battery_left, 8), (width - 8, 20), (190, 190, 190), 1)
    cv2.rectangle(frame, (width - 7, 11), (width - 5, 17), (190, 190, 190), -1)
    fill_width = int(25 * battery_level)
    if fill_width > 0:
        cv2.rectangle(
            frame,
            (battery_left + 2, 10),
            (battery_left + 2 + fill_width, 18),
            battery_color,
            -1,
        )

    if estimated_tower_pose is None:
        coordinate_text = "POS --   SHOW A TOWER MARKER"
    else:
        coordinate_text = (
            f"{pose_source}  "
            f"X {estimated_tower_pose['display_tower_x_cm']:.0f}  "
            f"Y {estimated_tower_pose['tower_y_cm']:.0f}  "
            f"Z {estimated_tower_pose['tower_z_cm']:.0f}  "
            f"H {estimated_tower_pose['heading_deg']:.0f}"
        )
    _put_fitted_text(frame, coordinate_text, (10, 48), width - 20, 0.48, pose_color, 1)

    action_top = height - 57
    cv2.rectangle(frame, (0, action_top), (4, height), action_color, -1)
    cv2.putText(
        frame,
        "ACTION",
        (10, action_top + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (170, 180, 190),
        1,
        cv2.LINE_AA,
    )
    _put_fitted_text(
        frame,
        action_title,
        (63, action_top + 16),
        width - 73,
        0.5,
        action_color,
        1,
    )
    _put_fitted_text(
        frame,
        action_detail,
        (10, height - 10),
        width - 20,
        0.42,
        (235, 238, 242),
        1,
    )


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
    target_takeoff_up_cm = 50
    current_target_index = 0
    inspection_done = False
    returning_home = False
    moving_to_next_face = False
    transition_mode = "SCAN"
    transition_phase = "SCAN"
    transition_waypoints = []
    transition_waypoint_index = 0
    acquire_start_ts = None
    battery_percent = None
    last_battery_update_ts = 0.0

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
        "last_estimation_cmd": (0, 0, 0, 0),
        "last_cmd_ts": 0.0,
        "last_pid_ts": None,
        "last_seen_ts": None,
        "last_seen_error_x": 0,
        "last_seen_error_y": 0,
        "last_seen_marker_id": None,
        "last_tower_pose": None,
        "waypoint_key": None,
        "waypoint_start_ts": None,
        "lock_accumulated_sec": 0.0,
        "lock_last_sample_ts": None,
        "lock_last_valid_ts": None,
        "photo_taken": False,
        "pids": {
            "x": PIDController(kp=0.35, kd=0.02),
            "y": PIDController(kp=0.45, kd=0.02),
            "z": PIDController(kp=0.35, kd=0.02),
            "yaw": PIDController(kp=0.7, kd=0.03),
            "waypoint_yaw": PIDController(
                kp=0.7,
                kd=0.03,
                output_limit=TOWER_WAYPOINT_MAX_YAW_SPEED,
            ),
        },
    }
    pose_estimator = {
        "initialized": False,
        "tower_x_cm": 0.0,
        "tower_y_cm": 0.0,
        "tower_z_cm": 0.0,
        "heading_deg": 0.0,
        "last_update_ts": None,
        "last_marker_ts": None,
        "last_marker_id": None,
    }

    try:
        Tello.FRAME_GRAB_TIMEOUT = 15
        tello.connect(wait_for_state=False)
        battery = tello.query_battery()
        battery_percent = int(battery)
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
            if returning_home:
                current_target_id = HOME_MARKER_ID
            else:
                safe_target_index = min(
                    current_target_index,
                    len(TARGET_MARKER_IDS) - 1,
                )
                current_target_id = TARGET_MARKER_IDS[safe_target_index]
            raw_frame = frame_reader.frame
            if raw_frame is None:
                if should_exit():
                    print("Exit key detected, shutting down...")
                    break
                time.sleep(0.02)
                continue

            now = time.monotonic()
            if now - last_battery_update_ts >= 1.0:
                try:
                    telemetry_battery = tello.get_current_state().get("bat")
                    if telemetry_battery is not None:
                        battery_percent = int(telemetry_battery)
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass
                last_battery_update_ts = now

            # 修正顏色通道：避免黃變藍 (RGB -> BGR)
            if FRAME_IS_RGB:
                raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

            # clean_frame: 儲存用乾淨高畫質影像 (無任何疊字)
            clean_frame = raw_frame.copy()

            # display_frame: 顯示/偵測用
            frame = cv2.resize(raw_frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            fallback_detection_frame = frame.copy()
            detection = _detect_aruco_detail(frame, current_target_id)
            marker_id = detection[2]
            if marker_id is None:
                preferred_id = pose_estimator.get("last_marker_id")
                if preferred_id is not None and preferred_id != current_target_id:
                    preferred_frame = fallback_detection_frame.copy()
                    preferred_detection = _detect_aruco_detail(
                        preferred_frame,
                        preferred_id,
                    )
                    if preferred_detection[2] is not None:
                        detection = preferred_detection
                        frame = preferred_frame

                if detection[2] is None:
                    detection = _detect_aruco_detail(fallback_detection_frame)
                    frame = fallback_detection_frame

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
                pose_rvec,
                pose_tvec,
            ) = detection
            tower_pose = estimate_tower_pose(
                marker_id,
                pose_rvec,
                pose_tvec,
            )
            if tower_pose is not None and not marker_allowed_for_localization(
                marker_id,
                current_target_id,
                moving_to_next_face,
                transition_mode,
            ):
                marker_id = None
                tower_pose = None
            loop_now = time.monotonic()
            predict_tower_pose(
                pose_estimator,
                tracker_state["last_estimation_cmd"],
                loop_now,
            )
            if tower_pose is not None:
                correct_tower_pose_estimate(
                    pose_estimator,
                    tower_pose,
                    marker_id,
                    loop_now,
                )
            estimated_tower_pose = get_estimated_tower_pose(pose_estimator)
            estimate_age = dead_reckoning_age(pose_estimator, loop_now)

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

            # HUD 資料：marker 可見時顯示絕對定位，否則顯示 RC 航位推算。
            if estimated_tower_pose is not None:
                if tower_pose is not None:
                    pose_source = f"M{marker_id}"
                    pose_color = (0, 255, 0)
                else:
                    pose_source = f"DR {estimate_age:.1f}s"
                    pose_color = (0, 165, 255)
            else:
                pose_source = "--"
                pose_color = (80, 80, 255)

            mode_name = "AUTO" if auto_track else "MANUAL"
            if returning_home:
                face_text = "HOME"
            elif inspection_done:
                face_text = "DONE"
            else:
                face_text = (
                    f"{current_target_index + 1}/{len(TARGET_MARKER_IDS)}"
                )
            search_text = transition_phase if moving_to_next_face else "SCAN"
            if transition_phase == "NAVIGATE" and transition_waypoints:
                search_text = (
                    f"NAV {transition_waypoint_index + 1}/"
                    f"{len(transition_waypoints)}"
                )
            progress_text = f"{face_text} {search_text}"
            action_title = "READY FOR TAKEOFF"
            action_detail = "Drone connected - awaiting takeoff"
            action_color = (90, 220, 120)
            if is_flying and not auto_track:
                action_title = "MANUAL CONTROL"
                action_detail = "Pilot has control"
                action_color = (80, 190, 255)
            elif inspection_done:
                action_title = "MISSION COMPLETE"
                action_detail = "Inspection finished"
                action_color = (90, 220, 120)

            real_marker_id = marker_id
            navigating_with_dead_reckoning = False
            if (
                marker_id is None
                and moving_to_next_face
                and transition_phase == "NAVIGATE"
                and estimated_tower_pose is not None
                and estimate_age <= DEAD_RECKONING_MAX_CONTROL_SEC
            ):
                marker_id = -1
                tower_pose = estimated_tower_pose
                error_x = None
                error_y = None
                navigating_with_dead_reckoning = True
            elif (
                marker_id is not None
                and marker_id != current_target_id
                and not (
                    moving_to_next_face
                    and transition_phase == "NAVIGATE"
                )
            ):
                # 非目標 marker 只更新座標，不交給拍照追蹤器。
                marker_id = None

            if is_flying and auto_track and not inspection_done:
                if marker_id is not None:
                    if not navigating_with_dead_reckoning:
                        tracker_state["last_seen_ts"] = time.monotonic()
                        tracker_state["last_seen_error_x"] = error_x
                        tracker_state["last_seen_error_y"] = error_y
                        tracker_state["last_seen_marker_id"] = marker_id
                        tracker_state["last_tower_pose"] = tower_pose
                    using_tower_waypoint = (
                        moving_to_next_face
                        and transition_phase == "NAVIGATE"
                    )
                    if using_tower_waypoint:
                        if transition_waypoints:
                            safe_waypoint_index = min(
                                transition_waypoint_index,
                                len(transition_waypoints) - 1,
                            )
                            target_position = transition_waypoints[safe_waypoint_index]
                        else:
                            target_position = tower_marker_target_position(current_target_id)
                        waypoint_errors, waypoint_cmd, waypoint_reached = fly_tower_waypoint(
                            tello,
                            tower_pose,
                            target_position,
                            tracker_state,
                        )
                        if waypoint_errors is None:
                            action_title = "WAITING FOR POSITION"
                            action_detail = "Show a valid tower marker"
                            action_color = (80, 80, 255)
                        else:
                            (
                                err_x,
                                err_y,
                                err_z,
                                body_lr,
                                body_fb,
                                heading_error,
                            ) = waypoint_errors
                            localization_text = (
                                "DR"
                                if navigating_with_dead_reckoning
                                else f"M{real_marker_id}"
                            )
                            waypoint_label = target_position.get("label", "WAYPOINT")
                            if returning_home:
                                action_title = f"RETURN HOME - {waypoint_label}"
                            elif waypoint_label == "BRIDGE TURN":
                                action_title = "TURN LEFT TO TOWER 2"
                            elif waypoint_label == "BRIDGE RIGHT":
                                action_title = "MOVE RIGHT TO TOWER 1"
                            elif waypoint_label.startswith("ARC"):
                                action_title = f"CHANGE FACE - {waypoint_label}"
                            elif waypoint_label == "FACE":
                                action_title = f"APPROACH ID {current_target_id}"
                            elif waypoint_label == "SAME":
                                action_title = f"MOVE TO ID {current_target_id}"
                            else:
                                action_title = f"NAVIGATE - {waypoint_label}"
                            action_detail = (
                                f"{localization_text}  ERR X {err_x:.0f}  "
                                f"Y {err_y:.0f}  Z {err_z:.0f}  H {heading_error:.0f}"
                            )
                            action_color = (255, 210, 70)
                        if waypoint_reached:
                            action_title = "WAYPOINT REACHED"
                            action_detail = (
                                f"Next step for ID {current_target_id}"
                            )
                            action_color = (90, 220, 120)
                            transition_waypoint_index += 1
                            reset_tracker_state(tracker_state)
                            if transition_waypoint_index < len(transition_waypoints):
                                tello.send_rc_control(0, 0, 0, 0)
                            else:
                                transition_phase = "ACQUIRE"
                                acquire_start_ts = time.monotonic()
                                tracker_state["last_seen_ts"] = None
                                tracker_state["last_seen_marker_id"] = None
                                search_for_transition_target(
                                    tello,
                                    transition_mode,
                                    acquire_start_ts,
                                    tracker_state,
                                )
                    else:
                        moving_to_next_face = False
                        transition_mode = "SCAN"
                        transition_phase = "SCAN"
                        transition_waypoints = []
                        transition_waypoint_index = 0
                        acquire_start_ts = None
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
                                action_title = "WAITING FOR MARKER POSE"
                                action_detail = f"Target ID {current_target_id}"
                                action_color = (80, 80, 255)
                            else:
                                body_lr_cm, ey_cm, body_fb_cm, eyaw_deg = pose_errors
                                if abs(error_y) > XY_LOCK_TOLERANCE:
                                    action_title = "ALIGN HEIGHT"
                                elif abs(error_x) >= X_EDGE_GUARD:
                                    action_title = "KEEP MARKER IN VIEW"
                                else:
                                    action_title = (
                                        "ALIGN HOME POSITION"
                                        if returning_home
                                        else f"ALIGN ID {current_target_id} FOR PHOTO"
                                    )
                                action_detail = (
                                    f"LR {body_lr_cm:.0f}  Y {ey_cm:.0f}  "
                                    f"FB {body_fb_cm:.0f}  YAW {eyaw_deg:.0f}"
                                )
                                action_color = (80, 220, 255)
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
                            action_title = f"ALIGN ID {current_target_id} FOR PHOTO"
                            action_detail = (
                                f"AREA {filtered_area}  DIST {dist_error}  "
                                f"YAW {'OK' if yaw_ok else 'ADJUST'}"
                            )
                            action_color = (80, 220, 255)

                        # 只累積有效鎖定時間；短暫漏偵測由下方 grace 邏輯保留。
                        if is_locked:
                            lock_now = time.monotonic()
                            last_sample_ts = tracker_state["lock_last_sample_ts"]
                            if last_sample_ts is not None:
                                tracker_state["lock_accumulated_sec"] += min(
                                    lock_now - last_sample_ts,
                                    0.1,
                                )
                            tracker_state["lock_last_sample_ts"] = lock_now
                            tracker_state["lock_last_valid_ts"] = lock_now
                            hold_time = tracker_state["lock_accumulated_sec"]
                            action_title = (
                                "LOCK HOME FOR LANDING"
                                if returning_home
                                else "HOLD STILL FOR PHOTO"
                            )
                            action_detail = (
                                f"LOCK {hold_time:.1f} / {PHOTO_LOCK_SECONDS:.1f} SEC"
                            )
                            action_color = (90, 220, 120)
                            if (
                                returning_home
                                and hold_time >= PHOTO_LOCK_SECONDS
                            ):
                                remember_rc_command(
                                    tracker_state,
                                    (0, 0, 0, 0),
                                )
                                tello.send_rc_control(0, 0, 0, 0)
                                print("Home marker locked. Landing...")
                                tello.land()
                                is_flying = False
                                auto_track = False
                                inspection_done = True
                                returning_home = False
                                moving_to_next_face = False
                                transition_mode = "SCAN"
                                transition_phase = "DONE"
                                transition_waypoints = []
                                transition_waypoint_index = 0
                                acquire_start_ts = None
                                reset_tracker_state(tracker_state)
                                mode_name = "MANUAL"
                                progress_text = "DONE"
                                action_title = "LANDED AT HOME"
                                action_detail = "Inspection complete"
                                action_color = (90, 220, 120)
                                print("Inspection complete. Returned home and landed.")
                            elif (
                                not tracker_state["photo_taken"]
                                and hold_time >= PHOTO_LOCK_SECONDS
                            ):
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
                                    returning_home = True
                                    reset_tracker_state(tracker_state)
                                    tracker_state["photo_taken"] = False
                                    moving_to_next_face = True
                                    transition_phase = "NAVIGATE"
                                    transition_waypoints = (
                                        build_tower_transition_waypoints(
                                            current_target_id,
                                            HOME_MARKER_ID,
                                        )
                                    )
                                    transition_waypoint_index = 0
                                    tracker_state["waypoint_key"] = None
                                    tracker_state["waypoint_start_ts"] = None
                                    acquire_start_ts = None
                                    transition_mode = "RETURN_HOME"
                                    print(
                                        "Inspection photos complete. "
                                        "Returning to ID 0 for landing."
                                    )
                                else:
                                    next_target_id = TARGET_MARKER_IDS[current_target_index]
                                    reset_tracker_state(tracker_state)
                                    tracker_state["photo_taken"] = False
                                    moving_to_next_face = True
                                    transition_phase = "NAVIGATE"
                                    transition_waypoints = build_tower_transition_waypoints(
                                        current_target_id,
                                        next_target_id,
                                    )
                                    transition_waypoint_index = 0
                                    tracker_state["waypoint_key"] = None
                                    tracker_state["waypoint_start_ts"] = None
                                    acquire_start_ts = None
                                    transition_mode = TRANSITION_MODES[
                                        current_target_index - 1
                                    ]
                                    print(
                                        f"Moving to target ID: {next_target_id} "
                                        f"mode: {transition_mode} "
                                        f"waypoints: {len(transition_waypoints)}"
                                    )
                        else:
                            tracker_state["lock_accumulated_sec"] = 0.0
                            tracker_state["lock_last_sample_ts"] = None
                            tracker_state["lock_last_valid_ts"] = None
                else:
                    lock_last_valid_ts = tracker_state.get("lock_last_valid_ts")
                    lock_loss_age = (
                        float("inf")
                        if lock_last_valid_ts is None
                        else time.monotonic() - lock_last_valid_ts
                    )
                    holding_lock_grace = (
                        not moving_to_next_face
                        and lock_last_valid_ts is not None
                        and lock_loss_age <= PHOTO_LOCK_LOSS_GRACE_SEC
                    )
                    if holding_lock_grace:
                        tracker_state["lock_last_sample_ts"] = time.monotonic()
                        remember_rc_command(tracker_state, (0, 0, 0, 0))
                        tello.send_rc_control(0, 0, 0, 0)
                        action_title = "MARKER FLICKER - HOLDING"
                        action_detail = (
                            f"Grace {lock_loss_age:.1f} / "
                            f"{PHOTO_LOCK_LOSS_GRACE_SEC:.1f} sec"
                        )
                        action_color = (40, 190, 255)
                    elif moving_to_next_face and transition_phase == "ACQUIRE":
                        if acquire_start_ts is None:
                            acquire_start_ts = time.monotonic()
                        search_cmd = search_for_transition_target(
                            tello,
                            transition_mode,
                            acquire_start_ts,
                            tracker_state,
                        )
                        if transition_mode == "DOWN":
                            action_title = f"SEARCH BELOW FOR ID {current_target_id}"
                        elif transition_mode == "UP":
                            action_title = f"SEARCH ABOVE FOR ID {current_target_id}"
                        elif transition_mode == "BRIDGE_RIGHT":
                            action_title = f"MOVE RIGHT - FIND ID {current_target_id}"
                        elif returning_home:
                            action_title = "TURN LEFT - FIND HOME ID 0"
                        else:
                            action_title = f"TURN LEFT - FIND ID {current_target_id}"
                        action_detail = f"ACQUIRE  RC {search_cmd}"
                        action_color = (40, 190, 255)
                    elif moving_to_next_face and transition_phase == "NAVIGATE":
                        search_cmd = continue_navigation_blind(
                            tello,
                            tracker_state,
                            transition_mode,
                        )
                        action_title = "NO MARKER - CONTINUE SAFELY"
                        action_detail = (
                            f"WP {transition_waypoint_index + 1}  RC {search_cmd}"
                        )
                        action_color = (40, 190, 255)
                    else:
                        reacquiring, reacquire_cmd = reacquire_lost_marker(
                            tello,
                            tracker_state,
                            DISPLAY_WIDTH,
                            DISPLAY_HEIGHT,
                        )
                        if reacquiring:
                            action_title = (
                                f"RECOVER ID {tracker_state['last_seen_marker_id']}"
                            )
                            action_detail = f"Last seen direction  RC {reacquire_cmd}"
                            action_color = (40, 190, 255)
                        else:
                            remember_rc_command(
                                tracker_state,
                                (0, 0, 0, SEARCH_YAW_SPEED),
                            )
                            tello.send_rc_control(0, 0, 0, SEARCH_YAW_SPEED)
                            action_title = f"SEARCH FOR ID {current_target_id}"
                            action_detail = f"Rotate left  RC yaw {SEARCH_YAW_SPEED}"
                            action_color = (40, 190, 255)

                    if not holding_lock_grace:
                        # 持續遺失才取消鎖定並重新尋找 marker。
                        reset_tracker_state(tracker_state, preserve_command=True)

            elif is_flying and auto_track and inspection_done:
                remember_rc_command(tracker_state, (0, 0, 0, 0))
                tello.send_rc_control(0, 0, 0, 0)

            elif is_flying and not auto_track:
                # 手動模式未按鍵時懸停
                remember_rc_command(tracker_state, (0, 0, 0, 0))
                tello.send_rc_control(0, 0, 0, 0)

            draw_flight_hud(
                frame,
                mode_name,
                progress_text,
                current_target_id,
                battery_percent,
                estimated_tower_pose,
                pose_source,
                pose_color,
                action_title,
                action_detail,
                action_color,
            )
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
                    returning_home = False
                    moving_to_next_face = False
                    transition_mode = "SCAN"
                    transition_phase = "SCAN"
                    transition_waypoints = []
                    transition_waypoint_index = 0
                    acquire_start_ts = None
                    reset_tracker_state(tracker_state)
                    tracker_state["photo_taken"] = False
                    print("Takeoff complete. Auto mode enabled.")

            # m: 自動/手動切換
            elif key == ord('m'):
                if is_flying:
                    auto_track = not auto_track
                    if not auto_track:
                        returning_home = False
                        moving_to_next_face = False
                        transition_mode = "SCAN"
                        transition_phase = "SCAN"
                        transition_waypoints = []
                        transition_waypoint_index = 0
                        acquire_start_ts = None
                    remember_rc_command(tracker_state, (0, 0, 0, 0))
                    tello.send_rc_control(0, 0, 0, 0)
                    print(f"Switched to {'AUTO' if auto_track else 'MANUAL'} mode.")

            # 手動操作（僅手動模式）
            elif is_flying and not auto_track and key == ord('w'):
                pulse_rc(tello, 0, 24, 0, 0, tracker_state, pose_estimator)
            elif is_flying and not auto_track and key == ord('s'):
                pulse_rc(tello, 0, -24, 0, 0, tracker_state, pose_estimator)
            elif is_flying and not auto_track and key == ord('a'):
                pulse_rc(tello, -24, 0, 0, 0, tracker_state, pose_estimator)
            elif is_flying and not auto_track and key == ord('d'):
                pulse_rc(tello, 24, 0, 0, 0, tracker_state, pose_estimator)
            elif is_flying and not auto_track and key == ord('r'):
                pulse_rc(tello, 0, 0, 24, 0, tracker_state, pose_estimator)
            elif is_flying and not auto_track and key == ord('f'):
                pulse_rc(tello, 0, 0, -24, 0, tracker_state, pose_estimator)
            elif is_flying and not auto_track and key == ord('z'):
                pulse_rc(tello, 0, 0, 0, -28, tracker_state, pose_estimator)
            elif is_flying and not auto_track and key == ord('c'):
                pulse_rc(tello, 0, 0, 0, 28, tracker_state, pose_estimator)

            # l: 降落
            elif key == ord('l'):
                auto_track = False
                returning_home = False
                moving_to_next_face = False
                transition_mode = "SCAN"
                transition_phase = "SCAN"
                transition_waypoints = []
                transition_waypoint_index = 0
                acquire_start_ts = None
                remember_rc_command(tracker_state, (0, 0, 0, 0))
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
