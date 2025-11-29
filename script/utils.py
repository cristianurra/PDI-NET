import math
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Optional, Generator

from config import ConfiguracionGlobal

def dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def depth_to_color(depth_cm: float, config: ConfiguracionGlobal) -> Tuple[int, int, int]:
    if depth_cm <= 0 or depth_cm >= config.MAX_DEPTH_CM:
        return (255, 0, 0)

    normalized_depth = np.clip(
        (depth_cm - config.MIN_DEPTH_CM) / (config.MAX_DEPTH_CM - config.MIN_DEPTH_CM),
        0.0,
        1.0
    )

    R = int(255 * (1 - normalized_depth))
    B = int(255 * normalized_depth)
    G = 0

    return (B, G, R)

def normalize_cell_view(current_image: np.ndarray, cell_target_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    try:
        normalized_image = cv2.resize(current_image, cell_target_size, interpolation=cv2.INTER_AREA)
        return normalized_image
    except Exception:
        return current_image

def register_image_to_map(current_image: np.ndarray, existing_image: np.ndarray) -> np.ndarray:
    if existing_image is None or current_image is None or existing_image.shape != current_image.shape:
        return current_image

    try:
        fused_image = cv2.addWeighted(existing_image, 0.5, current_image, 0.5, 0)
        return fused_image

    except Exception:
        return existing_image

def map_trans(hist_m: List[Tuple[float, float]], m_w: int, m_h: int, config: ConfiguracionGlobal) -> Tuple[float, float, float]:
    if not hist_m:
        return config.MAP_ESC_V, m_w / 2, m_h / 2

    xs, ys = [p[0] for p in hist_m], [p[1] for p in hist_m]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    r_x, r_y = max_x - min_x, max_y - min_y

    sz_disp = m_w - 2 * config.MAP_PAD_PX
    r_max = max(r_x, r_y)

    esc_m = sz_disp / r_max if r_max > 0 else config.MAP_ESC_V
    esc_m = min(esc_m, config.MAP_ESC_V)

    c_x, c_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    off_x = m_w / 2 - (c_x * esc_m)
    off_y = m_h / 2 + (c_y * esc_m)

    if r_x == 0 and r_y == 0:
        off_x, off_y = m_w / 2, m_h / 2
        esc_m = config.MAP_ESC_V

    return esc_m, off_x, off_y

def open_svo_file(svo_path: str) -> Tuple[Optional[Generator[Tuple[bool, Optional[np.ndarray]], None, None]], int, int, int]:
    try:
        import pyzed.sl as sl
    except ImportError:
        return None, 0, 0, 0

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        return None, 0, 0, 0

    total_frames = zed.get_svo_number_of_frames()
    cam_info = zed.get_camera_information()
    w = cam_info.camera_resolution.width * 2
    h = cam_info.camera_resolution.height

    def frame_generator():
        left_image = sl.Mat()
        right_image = sl.Mat()

        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                zed.retrieve_image(right_image, sl.VIEW.RIGHT)

                left_bgr = left_image.get_data()[:, :, :3]
                right_bgr = right_image.get_data()[:, :, :3]

                stereo_frame = np.hstack((left_bgr, right_bgr))

                yield True, stereo_frame
            else:
                yield False, None
                zed.close()
                break

    return frame_generator(), total_frames, w, h
