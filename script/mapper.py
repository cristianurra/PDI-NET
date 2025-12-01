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
        # Necesitamos al menos 1 punto con historia para movernos (fallback)
        # O 3 puntos para el cálculo complejo (affine)
        if len(tracked_objects) < 1: return

        prev_pts = []
        curr_pts = []

        for obj in tracked_objects:
            # Verificamos que el objeto tenga historia suficiente
            if len(obj.get('hist_vel', [])) > 0 and len(obj.get('hist_pos', [])) >= 2:
                curr = obj['pos']
                prev = obj['hist_pos'][-2]
                
                prev_pts.append(prev)
                curr_pts.append(curr)

        if len(prev_pts) < 1: return

        prev_np = np.array(prev_pts, dtype=np.float32).reshape(-1, 1, 2)
        curr_np = np.array(curr_pts, dtype=np.float32).reshape(-1, 1, 2)

        # Variables para el movimiento calculado
        dx_local = 0.0
        dy_local = 0.0
        da_local = 0.0
        success = False

        # INTENTO 1: Estimación Afin (Precisa, detecta rotación)
        if len(prev_pts) >= 3:
            try:
                M_obj_to_prev, inliers = cv2.estimateAffinePartial2D(curr_np, prev_np, method=cv2.RANSAC)
                if M_obj_to_prev is not None:
                    dx_local = -M_obj_to_prev[0, 2] # Invertimos signo (movimiento cámara vs mundo)
                    dy_local = -M_obj_to_prev[1, 2]
                    da_local = np.arctan2(M_obj_to_prev[1, 0], M_obj_to_prev[0, 0])
                    success = True
            except Exception:
                success = False

        # INTENTO 2: Fallback de Centroides (Simple, si el anterior falla)
        if not success:
            # Simplemente calculamos el promedio de desplazamiento
            mean_prev = np.mean(prev_pts, axis=0)
            mean_curr = np.mean(curr_pts, axis=0)
            
            movement = mean_prev - mean_curr # Prev - Curr da el movimiento de la cámara sobre el fondo
            dx_local = movement[0]
            dy_local = movement[1]
            da_local = 0.0 # Asumimos sin rotación en fallback

        # Filtro de ruido excesivo (Teletransportación)
        dist_sq = dx_local**2 + dy_local**2
        if dist_sq > 2500: # Si se mueve más de 50px por frame, es error
            return

        # TRANSFORMACIÓN DE COORDENADAS (CÁMARA -> MUNDO)
        # El dx_local está alineado con la cámara. Debemos rotarlo por el ángulo actual del ROV.
        c = np.cos(self.global_angle)
        s = np.sin(self.global_angle)

        # Rotación 2D:
        # dx_global = dx_local * cos - dy_local * sin
        # dy_global = dx_local * sin + dy_local * cos
        
        # FACTOR DE ESCALA PIXEL A CM REAL (De config.py o estimado)
        # Usamos self.config.CM_POR_PX si está disponible, sino 1.0
        factor_cm = getattr(self.config, 'CM_POR_PX', 0.1) 

        dx_global_cm = (dx_local * c - dy_local * s) * factor_cm
        dy_global_cm = (dx_local * s + dy_local * c) * factor_cm

        # ACTUALIZACIÓN DE ESTADO
        self.global_x += dx_global_cm
        self.global_y += dy_global_cm
        self.global_angle += da_local * 0.5 # Suavizado de giro

        # Guardar trayectoria cada ciertos cm para no saturar
        last_pos = self.trayectoria[-1]
        dist_to_last = (self.global_x - last_pos[0])**2 + (self.global_y - last_pos[1])**2
        
        if dist_to_last > 5: # Solo guardar si se movió algo
            self.trayectoria.append((int(self.global_x), int(self.global_y)))


    def draw_map(self, tracked_objects: List[Dict[str, Any]], is_quality_good: bool = True, frames_history: List[List[Dict[str, Any]]] = None) -> np.ndarray:
        if frames_history is None:
            frames_history = []

        # --- AJUSTE DE DIRECCIÓN (CALIBRACIÓN) ---
        # Si el ROV va a la derecha y la línea sale a la izquierda, cambia DIR_X.
        # Si el ROV va adelante y la línea sale adelante (en vez de atrás), cambia DIR_Y.
        # Valores posibles: 1.0 o -1.0
        DIR_X = -1.0 
        DIR_Y = -1.0 

        # Escala
        CM_PER_MAP_PIXEL_LOCAL = 1.0 
        self.scale = 0.5 

        cx_screen = int(self.map_size / 2)
        cy_screen = int(self.map_size / 2)
        
        # Offset inicial para centrar
        START_OFFSET_X = 500.0
        START_OFFSET_Y = 500.0

        canvas = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

        # 1. GRILLA ESTÁTICA
        grid_step = int((100.0 / CM_PER_MAP_PIXEL_LOCAL) * self.scale)
        color_grid = (30, 30, 30)
        for i in range(0, self.map_size, grid_step):
            cv2.line(canvas, (i, 0), (i, self.map_size), color_grid, 1)
            cv2.line(canvas, (0, i), (self.map_size, i), color_grid, 1)
        cv2.line(canvas, (cx_screen, 0), (cx_screen, self.map_size), (50, 50, 50), 1)
        cv2.line(canvas, (0, cy_screen), (self.map_size, cy_screen), (50, 50, 50), 1)

        # 2. DIBUJAR TRAYECTORIA (APLICANDO DIR_X / DIR_Y)
        if len(self.trayectoria) > 1:
            pts_to_draw = []
            for pt in self.trayectoria:
                # Normalizar (restar inicio)
                raw_dx = pt[0] - START_OFFSET_X
                raw_dy = pt[1] - START_OFFSET_Y
                
                # APLICAR CORRECCIÓN DE DIRECCIÓN
                dx = raw_dx * DIR_X
                dy = raw_dy * DIR_Y
                
                # Proyección
                px = int(cx_screen + (dx / CM_PER_MAP_PIXEL_LOCAL) * self.scale)
                py = int(cy_screen - (dy / CM_PER_MAP_PIXEL_LOCAL) * self.scale)
                pts_to_draw.append([px, py])

            pts_np = np.array(pts_to_draw, np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_np], False, (0, 255, 255), 2)

        # 3. DIBUJAR ROV (APLICANDO DIR_X / DIR_Y)
        raw_cur_dx = self.global_x - START_OFFSET_X
        raw_cur_dy = self.global_y - START_OFFSET_Y
        
        # Posición corregida del ROV
        cur_dx = raw_cur_dx * DIR_X
        cur_dy = raw_cur_dy * DIR_Y
        
        rov_px = int(cx_screen + (cur_dx / CM_PER_MAP_PIXEL_LOCAL) * self.scale)
        rov_py = int(cy_screen - (cur_dy / CM_PER_MAP_PIXEL_LOCAL) * self.scale)
        
        # Rotación ROV
        c_rov = np.cos(self.global_angle)
        s_rov = np.sin(self.global_angle)
        
        # Triángulo
        pts_loc = [np.array([0, 20]), np.array([-10, -10]), np.array([10, -10])]
        vals_rov = []
        for p in pts_loc:
            xr = p[0] * c_rov - p[1] * s_rov
            yr = p[0] * s_rov + p[1] * c_rov
            vals_rov.append([int(rov_px + xr), int(rov_py - yr)])
            
        cv2.drawContours(canvas, [np.array(vals_rov, np.int32)], 0, (0, 0, 255), -1)
        
        # Nariz
        nose_base = np.array([0, 35])
        nose_x = nose_base[0] * c_rov - nose_base[1] * s_rov
        nose_y = nose_base[0] * s_rov + nose_base[1] * c_rov
        cv2.line(canvas, (rov_px, rov_py), (int(rov_px + nose_x), int(rov_py - nose_y)), (255, 0, 0), 2)

        # 4. PUNTOS DETECTADOS (APLICANDO DIR_X / DIR_Y)
        all_objects = tracked_objects 
        for obj in all_objects:
            z = obj.get('depth_cm', 0)
            if z > 10 and z < 800:
                u, v = obj['pos']
                cx_cam = 1280 / 2 
                x_local = (u - cx_cam) * z / self.config.FOCAL_PIX
                y_local = z 

                # Rotación local -> global
                x_rot = x_local * c_rov - y_local * s_rov
                y_rot = x_local * s_rov + y_local * c_rov
                
                # Posición absoluta corregida
                # Sumamos el desplazamiento al ROV actual (que ya tiene DIR aplicado)
                x_final_cm = cur_dx + x_rot
                y_final_cm = cur_dy + y_rot

                px = int(cx_screen + (x_final_cm / CM_PER_MAP_PIXEL_LOCAL) * self.scale)
                py = int(cy_screen - (y_final_cm / CM_PER_MAP_PIXEL_LOCAL) * self.scale)

                if 0 <= px < self.map_size and 0 <= py < self.map_size:
                    color = obj.get('color', (0, 255, 0))
                    color = (int(color[0]), int(color[1]), int(color[2]))
                    cv2.circle(canvas, (px, py), 2, color, -1)

        if is_quality_good:
            cv2.putText(canvas, f"X:{cur_dx:.0f} Y:{cur_dy:.0f}", (10, self.map_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return canvas
