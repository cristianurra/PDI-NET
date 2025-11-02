import math
import numpy as np
import cv2
from config import MIN_DEPTH_CM, MAX_DEPTH_CM, MAP_ESC_V, MAP_PAD_PX # ¡CORREGIDO!

def dist(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def depth_to_color(depth_cm):
    if depth_cm <= 0 or depth_cm >= MAX_DEPTH_CM:
        return (255, 0, 0)

    normalized_depth = np.clip(
        (depth_cm - MIN_DEPTH_CM) / (MAX_DEPTH_CM - MIN_DEPTH_CM),
        0.0,
        1.0
    )

    R = int(255 * (1 - normalized_depth))
    B = int(255 * normalized_depth)
    G = 0

    return (B, G, R)

def map_trans(hist_m, m_w, m_h):
    if not hist_m:
        return MAP_ESC_V, m_w / 2, m_h / 2

    xs, ys = [p[0] for p in hist_m], [p[1] for p in hist_m]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    r_x, r_y = max_x - min_x, max_y - min_y

    sz_disp = m_w - 2 * MAP_PAD_PX
    r_max = max(r_x, r_y)

    esc_m = sz_disp / r_max if r_max > 0 else MAP_ESC_V
    esc_m = min(esc_m, MAP_ESC_V)

    c_x, c_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    off_x = m_w / 2 - (c_x * esc_m)
    off_y = m_h / 2 + (c_y * esc_m)

    if r_x == 0 and r_y == 0:
        off_x, off_y = m_w / 2, m_h / 2
        esc_m = MAP_ESC_V

    return esc_m, off_x, off_y

def normalize_cell_view(current_image, cell_target_size=(100, 100)):
    try:
        normalized_image = cv2.resize(current_image, cell_target_size, interpolation=cv2.INTER_AREA)
        return normalized_image
    except Exception:
        return current_image

def register_image_to_map(current_image, existing_image):
    if existing_image is None or current_image is None or existing_image.shape != current_image.shape:
        return current_image

    try:
        fused_image = cv2.addWeighted(existing_image, 0.5, current_image, 0.5, 0)
        return fused_image

    except Exception:
        return existing_image
