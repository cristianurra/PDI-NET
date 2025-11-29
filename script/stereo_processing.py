import cv2
import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional

from config import ConfiguracionGlobal

def proc_seg(frame: np.ndarray, k_uni: np.ndarray, k_limp: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        src=blurred, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV, blockSize=15, C=1
    )

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_uni)
    filtered = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_limp)
    return filtered


def get_cns(cns_filt: np.ndarray, q_w: int, q_h: int, w: int, config: ConfiguracionGlobal, y_max_track: int = 9999) -> Tuple[List[Tuple[int, int]], List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], int]], List[Tuple[Tuple[int, int], int]]]:
    m_x = w // 2

    cns_all = []
    contours, hier = cv2.findContours(cns_filt, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        for i in range(len(contours)):
            if hier is not None and hier[0][i][3] != -1 and cv2.contourArea(contours[i]) >= 50:
                M = cv2.moments(contours[i])
                if M["m00"] != 0:
                    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                    q_col, q_row = cX // q_w, cY // q_h
                    if (q_row, q_col) in config.Q_ACT_BASE:
                        cns_all.append((cX, cY))

    cns_L = [c for c in cns_all if c[0] < m_x]
    cns_R = [c for c in cns_all if c[0] >= m_x]

    matched_cns_pairs = []
    used_R = [False] * len(cns_R)

    for cL in cns_L:
        xL, yL = cL
        mejor_match_i = -1

        for i, cR in enumerate(cns_R):
            if not used_R[i]:
                xR, yR = cR
                xR_adj = xR - m_x

                if abs(yL - yR) <= config.Y_TOLERANCE:
                    disparity = xL - xR_adj

                    if config.MIN_DISPARITY <= disparity <= config.MAX_DISPARITY:
                        mejor_match_i = i
                        break

        if mejor_match_i != -1:
            cR_matched = cns_R[mejor_match_i]
            used_R[mejor_match_i] = True
            xR_adj = cR_matched[0] - m_x
            disparity = xL - xR_adj
            matched_cns_pairs.append(((cL, cR_matched), disparity))

    cns_L_matched_only = [pair[0][0] for pair in matched_cns_pairs]
    cns_disp_only = [(pair[0][0], pair[1]) for pair in matched_cns_pairs]

    return cns_L_matched_only, matched_cns_pairs, cns_disp_only


def proc_mesh_mask(frame: np.ndarray, mesh_consolidate_k: int, k_limp: np.ndarray, k_vert_fill: np.ndarray) -> Optional[np.ndarray]:
    if frame is None:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    blurred_s = cv2.GaussianBlur(saturation, (3, 3), 0)

    binary_s = cv2.adaptiveThreshold(
        src=blurred_s, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV, blockSize=15, C=1
    )

    binary = binary_s

    kernel_sq = np.ones((mesh_consolidate_k, mesh_consolidate_k), np.uint8)
    eroded = cv2.erode(binary, kernel_sq, iterations=1)
    opened = cv2.dilate(eroded, kernel_sq, iterations=1)

    closed_vertical = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_vert_fill, iterations=1)

    filtered = cv2.morphologyEx(closed_vertical, cv2.MORPH_OPEN, k_limp, iterations=1)

    return filtered


def get_mesh_boundary_y_pos(consolidated_mask: Optional[np.ndarray], w_half: int, h_total: int, kernel_edge: np.ndarray) -> int:
    if consolidated_mask is None:
        return 0

    mask_left = consolidated_mask[:, :w_half].copy()
    if mask_left.dtype != np.uint8 or len(mask_left.shape) != 2:
        mask_left = cv2.convertScaleAbs(mask_left)

    _, mask_left = cv2.threshold(mask_left, 1, 255, cv2.THRESH_BINARY)

    eroded = cv2.erode(mask_left, kernel_edge, iterations=1)
    boundary_mask = cv2.subtract(mask_left, eroded)

    contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_y_positions = []
    roi_top_y = h_total // 2

    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            if area > 100:
                current_min_y = int(np.min(c[:, :, 1]))
                if current_min_y < roi_top_y:
                    valid_y_positions.append(current_min_y)

    if valid_y_positions:
        return int(np.median(valid_y_positions))

    return 0


def detect_orange_markers(bgr_image: np.ndarray, config: ConfiguracionGlobal) -> List[Dict[str, Any]]:
    if bgr_image is None:
        return []

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lower = np.array(config.ORANGE_HSV_LOW, dtype=np.uint8)
    upper = np.array(config.ORANGE_HSV_HIGH, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < config.ORANGE_MIN_AREA or area > config.ORANGE_MAX_AREA:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circ = 4.0 * math.pi * area / (perimeter * perimeter)

        if circ < config.ORANGE_CIRCULARITY:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        x, y, w_box, h_box = cv2.boundingRect(c)

        results.append({'cx': cX[0], 'cy': cX[1], 'area': area, 'circ': circ, 'bbox': (x, y, w_box, h_box)})

    return results
