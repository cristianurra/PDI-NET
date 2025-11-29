import numpy as np
import cv2
import math
from typing import List, Tuple, Dict, Any

from config import ConfiguracionGlobal

class GlobalMapper2D:
    def __init__(self, config: ConfiguracionGlobal, center_x: float = 500.0, center_y: float = 500.0):
        self.map_size = 1000
        self.global_x = center_x
        self.global_y = center_y
        self.global_angle = 0.0
        self.trayectoria: List[Tuple[int, int]] = [(int(center_x), int(center_y))]
        self.scale = 2.0
        self.config = config

    def update_position(self, tracked_objects: List[Dict[str, Any]]):
        if len(tracked_objects) < 3: return

        prev_pts = []
        curr_pts = []

        for obj in tracked_objects:
            if len(obj['hist_vel']) > 0 and len(obj['hist_pos']) >= 2:
                curr = obj['pos']
                prev = obj['hist_pos'][-2]

                prev_pts.append(prev)
                curr_pts.append(curr)

        if len(prev_pts) < 3: return

        prev_np = np.array(prev_pts, dtype=np.float32).reshape(-1, 1, 2)
        curr_np = np.array(curr_pts, dtype=np.float32).reshape(-1, 1, 2)

        M_obj_to_prev, inliers = cv2.estimateAffinePartial2D(curr_np, prev_np, method=cv2.RANSAC)

        if M_obj_to_prev is not None:

            dx_malla_local = -M_obj_to_prev[0, 2]
            dy_malla_local = -M_obj_to_prev[1, 2]

            d_angle_malla = np.arctan2(M_obj_to_prev[1, 0], M_obj_to_prev[0, 0])

            d_angle_cam = -d_angle_malla

            dist_sq = dx_malla_local**2 + dy_malla_local**2

            if dist_sq > 900:
                dx_malla_local = 0
                dy_malla_local = 0
                d_angle_cam = 0

            self.global_angle += d_angle_cam * 0.5

            self.trayectoria.append((int(self.global_x), int(self.global_y)))


    def draw_map(self, tracked_objects: List[Dict[str, Any]], is_quality_good: bool = True, frames_history: List[List[Dict[str, Any]]] = None) -> np.ndarray:

        if frames_history is None:
            frames_history = []

        CM_PER_MAP_PIXEL_LOCAL = 0.5

        self.scale = self.config.MAP_ZOOM_FACTOR

        cx_map_center = int(self.map_size / 2)
        cy_map_center = int(self.map_size / 2)

        offset_x = cx_map_center
        offset_y = cy_map_center


        canvas = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

        for i in range(0, self.map_size, 100):
            cv2.line(canvas, (i, 0), (i, self.map_size), (30, 30, 30), 1)
            cv2.line(canvas, (0, i), (i, self.map_size), (30, 30, 30), 1)

        for i in range(len(self.trayectoria) - 1):
            p1 = self.trayectoria[i]
            p2 = self.trayectoria[i+1]

            p1_map = (int(offset_x + (p1[0] - self.global_x) / CM_PER_MAP_PIXEL_LOCAL * self.scale),
                      int(offset_y + (p1[1] - self.global_y) / CM_PER_MAP_PIXEL_LOCAL * self.scale))
            p2_map = (int(offset_x + (p2[0] - self.global_x) / CM_PER_MAP_PIXEL_LOCAL * self.scale),
                      int(offset_y + (p2[1] - self.global_y) / CM_PER_MAP_PIXEL_LOCAL * self.scale))

            cv2.line(canvas, p1_map, p2_map, (0, 255, 255), 2)


        cx_map = int(self.map_size / 2)
        cy_map = int(self.map_size / 2)

        rov_size_factor = 0.1

        tip = np.array([0, -15]) * self.scale * rov_size_factor / CM_PER_MAP_PIXEL_LOCAL
        bl = np.array([-10, 10]) * self.scale * rov_size_factor / CM_PER_MAP_PIXEL_LOCAL
        br = np.array([10, 10]) * self.scale * rov_size_factor / CM_PER_MAP_PIXEL_LOCAL

        c, s = np.cos(self.global_angle), np.sin(self.global_angle)
        rot_mat = np.array([[c, -s], [s, c]])

        p1 = np.dot(rot_mat, tip) + [cx_map, cy_map]
        p2 = np.dot(rot_mat, bl) + [cx_map, cy_map]
        p3 = np.dot(rot_mat, br) + [cx_map, cy_map]

        pts_rov = np.array([p1, p2, p3], np.int32)
        cv2.drawContours(canvas, [pts_rov], 0, (0, 0, 255), -1)

        p_nose = np.dot(rot_mat, np.array([0, -25])) * self.scale * rov_size_factor / CM_PER_MAP_PIXEL_LOCAL + [cx_map, cy_map]
        cv2.line(canvas, (cx_map, cy_map), (int(p_nose[0]), int(p_nose[1])), (255, 0, 0), 2)

        for history_frame in frames_history:
            for obj in history_frame:
                u, v = obj['pos']
                z = obj.get('depth_cm', 0)
                if z > 0:
                    cx_L = 640/2

                    x_cam_cm = (u - cx_L) * z / self.config.FOCAL_PIX

                    local_x_map_px = x_cam_cm / CM_PER_MAP_PIXEL_LOCAL
                    local_y_map_px = -z / CM_PER_MAP_PIXEL_LOCAL

                    world_x_map_px = local_x_map_px * c - local_y_map_px * s
                    world_y_map_px = local_x_map_px * s + local_y_map_px * c

                    wx = int(cx_map + world_x_map_px * self.scale)
                    wy = int(cy_map + world_y_map_px * self.scale)

                    if 0 <= wx < self.map_size and 0 <= wy < self.map_size:
                        color = obj.get('color', (200,200,200))
                        cv2.circle(canvas, (wx, wy), 2, color, -1)

        if is_quality_good:
            cv2.putText(canvas, f"ANG: {math.degrees(self.global_angle):.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(canvas, "ESPERANDO DATOS...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return canvas
