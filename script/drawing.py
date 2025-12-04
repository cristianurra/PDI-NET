"""
Módulo de funciones de visualización y dibujo.
Contiene funciones para renderizar elementos visuales como puntos, vectores,
mapas, escalas de profundidad y otros elementos gráficos del sistema.
"""

import cv2
import numpy as np
import random
import math
from typing import List, Tuple, Dict, Any

from config import ConfiguracionGlobal
from utils import depth_to_color, map_trans

def dib_escala_profundidad(frame: np.ndarray, w: int, h: int, config: ConfiguracionGlobal):
    """
    Dibuja una barra de escala de profundidad en el frame.
    Muestra un gradiente de color que representa la profundidad (cerca=rojo, lejos=azul).
    
    Args:
        frame: Imagen BGR donde dibujar la escala
        w: Ancho del frame
        h: Alto del frame
        config: Configuración con rangos MIN_DEPTH_CM y MAX_DEPTH_CM
    """
    BAR_W, BAR_H = 50, 300
    BAR_X = w - BAR_W - 20
    BAR_Y = max(10, h // 2 - BAR_H // 2)

    bar_img = np.zeros((BAR_H, BAR_W, 3), dtype=np.uint8)
    for i in range(BAR_H):
        norm_val = 1.0 - (i / BAR_H)
        depth_cm = config.MIN_DEPTH_CM + norm_val * (config.MAX_DEPTH_CM - config.MIN_DEPTH_CM)
        color = depth_to_color(depth_cm, config)
        bar_img[i, :, :] = color

    y_end = min(h, BAR_Y + BAR_H)
    x_end = min(w, BAR_X + BAR_W)
    if BAR_Y < y_end and BAR_X < x_end:
        frame[BAR_Y:y_end, BAR_X:x_end] = bar_img[0:(y_end - BAR_Y), 0:(x_end - BAR_X)]


def dib_mov(frame: np.ndarray, objs: List[Dict[str, Any]], w: int, h: int, depth_cm: float, config: ConfiguracionGlobal, mostrar_vector: bool = True) -> Tuple[float, float, np.ndarray]:
    """
    Dibuja objetos trackeados y sus vectores de movimiento en el frame.
    Calcula velocidad promedio y retorna imagen limpia de la zona izquierda.
    
    Args:
        frame: Imagen BGR donde dibujar
        objs: Lista de objetos trackeados
        w: Ancho del frame
        h: Alto del frame
        depth_cm: Profundidad promedio actual
        config: Configuración global
        mostrar_vector: Si True, dibuja vectores de movimiento
    
    Returns:
        Tupla de (velocidad_x, velocidad_y, imagen_limpia)
    """
    vels_t = []

    objs_estables = [obj for obj in objs if obj['supervivencia_fr'] >= config.MIN_SUPERVIVENCIA_FR]

    num_mos = max(1, int(len(objs_estables) * config.PORC_MOS)) if objs_estables else 0
    objs_dib = random.sample(objs_estables, num_mos) if num_mos > 0 else []

    clean_area_image = frame[:, :w // 2].copy()

    for obj in objs_dib:
        cX, cY = obj['pos']
        cX_R, cY_R = obj['pos_R']
        color_punto = obj['color']

        cv2.circle(frame, (cX, cY), config.RAD_PUN, color_punto, -1)
        cv2.circle(frame, (cX_R, cY_R), config.RAD_PUN, color_punto, -1)

        v_x = np.median([v[0] for v in obj['hist_vel']])
        v_y = np.median([v[1] for v in obj['hist_vel']])

        if abs(v_x) > 1 or abs(v_y) > 1:
            vels_t.append((v_x, v_y))

    def dib_vec(vels: List[Tuple[float, float]], c_x: int, c_y: int, color: Tuple[int, int, int], lbl: str = ""):
        if vels:
            CIRCLE_RADIUS = 80
            MAX_VECTOR_LENGTH = CIRCLE_RADIUS * 0.8

            cv2.circle(frame, (c_x, c_y), CIRCLE_RADIUS, config.C_CAM, 6)

            m_vx = np.median([v[0] for v in vels])
            m_vy = np.median([v[1] for v in vels])

            magnitude = np.sqrt(m_vx**2 + m_vy**2)
            current_scaled_vx = m_vx * config.ESC_VEC
            current_scaled_vy = m_vy * config.ESC_VEC
            current_length = magnitude * config.ESC_VEC

            final_vx = current_scaled_vx
            final_fy = current_scaled_vy

            if magnitude > 0 and current_length > MAX_VECTOR_LENGTH:
                scale_factor = MAX_VECTOR_LENGTH / current_length
                final_vx = current_scaled_vx * scale_factor
                final_fy = current_scaled_vy * scale_factor

            p2_x = int(c_x - final_vx)
            p2_y = int(c_y - final_fy)

            cv2.arrowedLine(frame, (c_x, c_y), (p2_x, p2_y), color, 8, tipLength=0.4)


    v_cam_x, v_cam_y = 0.0, 0.0
    if vels_t:
        v_cam_x = float(np.median([v[0] for v in vels_t]))
        v_cam_y = float(np.median([v[1] for v in vels_t]))
        if mostrar_vector:
            # Dibujar en vista izquierda (siempre visible)
            dib_vec(vels_t, w // 4, h // 3, config.C_CAM, 'CAMARA')

    return -v_cam_x, -v_cam_y, clean_area_image


def dib_vector_yolo(frame: np.ndarray, w: int, h: int, velocity_x: float, velocity_y: float, config: ConfiguracionGlobal):
    """
    Dibuja el vector de odometría YOLO en la parte inferior de la vista mono.
    
    Args:
        frame: Frame sobre el que dibujar
        w: Ancho del frame
        h: Alto del frame
        velocity_x: Componente X de velocidad de la cámara
        velocity_y: Componente Y de velocidad de la cámara
        config: Configuración global
    """
    CIRCLE_RADIUS = 80
    MAX_VECTOR_LENGTH = CIRCLE_RADIUS * 0.8
    
    # Centro en la parte inferior
    c_x = w // 2
    c_y = int(h * 2 / 3)
    
    # Dibujar círculo
    cv2.circle(frame, (c_x, c_y), CIRCLE_RADIUS, (0, 255, 255), 5)  # Amarillo cian
    
    # Calcular vector
    magnitude = np.sqrt(velocity_x**2 + velocity_y**2)
    if magnitude > 0.5:  # Solo dibujar si hay movimiento significativo
        scaled_vx = velocity_x * config.ESC_VEC
        scaled_vy = velocity_y * config.ESC_VEC
        current_length = magnitude * config.ESC_VEC
        
        # Limitar longitud máxima
        if current_length > MAX_VECTOR_LENGTH:
            scale_factor = MAX_VECTOR_LENGTH / current_length
            scaled_vx *= scale_factor
            scaled_vy *= scale_factor
        
        # Dibujar flecha (velocidad ya está en dirección correcta de cámara)
        p2_x = int(c_x + scaled_vx)
        p2_y = int(c_y + scaled_vy)
        cv2.arrowedLine(frame, (c_x, c_y), (p2_x, p2_y), (0, 255, 255), 6, tipLength=0.4)
    
    # Etiqueta
    cv2.putText(frame, "YOLO", (c_x - 30, c_y + CIRCLE_RADIUS + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def dib_ayu(frame: np.ndarray, w: int, h: int, q_w: int, q_h: int, config: ConfiguracionGlobal):
    m_x = w // 2
    cv2.line(frame, (m_x, 0), (m_x, h), config.C_NARAN, 2)

    for col in range(1, config.Q_X):
        cv2.line(frame, (col * q_w, 0), (col * q_w, h), config.C_GRIS, 1)
    for row in range(1, config.Q_Y):
        cv2.line(frame, (0, row * q_h), (w, row * q_h), config.C_GRIS, 1)

    T_ACT = 3
    for row, col in config.Q_ACT_BASE:
        x1 = col * q_w
        y1 = row * q_h
        x2 = (col + 1) * q_w
        y2 = (row + 1) * q_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), config.C_ACT, T_ACT)


def dib_map(
    hist_celdas_vis: Dict[Tuple[int, int], Tuple[float, np.ndarray]],
    pos_m_x: float, pos_m_y: float,
    fixed_grid_sz_cm: float, rect_sz_cm_actual: float,
    map_w_display: int, map_h_display: int,
    current_view_w_cm: float, current_view_h_cm: float, config: ConfiguracionGlobal
) -> np.ndarray:
    celdas_xy_cm = list(hist_celdas_vis.keys())

    if not hist_celdas_vis:
        celdas_xy_cm = [(pos_m_x / fixed_grid_sz_cm, pos_m_y / fixed_grid_sz_cm)]

    celdas_xy_cm_normalized = [(x * fixed_grid_sz_cm, y * fixed_grid_sz_cm) for x, y in celdas_xy_cm]
    celdas_xy_cm_normalized.append((pos_m_x, pos_m_y))

    sz = max(map_w_display, map_h_display)
    canv_m = np.zeros((sz, sz, 3), dtype=np.uint8)

    esc_m, off_x, off_y = map_trans(celdas_xy_cm_normalized, sz, sz, config)

    for (grid_x, grid_y), data in hist_celdas_vis.items():

        depth_c, image_c = data

        cell_center_x = grid_x * fixed_grid_sz_cm
        cell_center_y = grid_y * fixed_grid_sz_cm

        rect_size_cm = fixed_grid_sz_cm

        rect_size_px = int((rect_size_cm - config.RECT_MARGIN_CM) * esc_m)

        map_center_x = int(off_x + cell_center_x * esc_m)
        map_center_y = int(off_y - cell_center_y * esc_m)  # Invertir Y para el mapa

        half_size_px = rect_size_px // 2
        y1 = map_center_y - half_size_px
        y2 = map_center_y + half_size_px
        x1 = map_center_x - half_size_px
        x2 = map_center_x + half_size_px

        y1_clip, y2_clip = max(0, y1), min(sz, y2)
        x1_clip, x2_clip = max(0, x1), min(sz, x2)

        target_h, target_w = y2_clip - y1_clip, x2_clip - x1_clip

        if target_h > 0 and target_w > 0:
            try:
                img_to_draw = cv2.resize(image_c, (target_w, target_h), interpolation=cv2.INTER_AREA)

                canv_m[y1_clip:y2_clip, x1_clip:x2_clip] = img_to_draw[
                    0:target_h,
                    0:target_w
                ]
            except cv2.error:
                fail_color = (150, 150, 150)
                cv2.rectangle(canv_m, (x1, y1), (x2, y2), fail_color, -1)

    cur_x = int(off_x + pos_m_x * esc_m)
    cur_y = int(off_y - pos_m_y * esc_m)  # Invertir Y para el mapa

    VIEW_RECT_SIZE_CM = fixed_grid_sz_cm * 0.9

    if current_view_w_cm > 0 and current_view_h_cm > 0:
        W_PX = int(VIEW_RECT_SIZE_CM * esc_m)
        H_PX = int(VIEW_RECT_SIZE_CM * esc_m)

        half_W_PX = W_PX // 2
        half_H_PX = H_PX // 2

        x1 = cur_x - half_W_PX
        y1 = cur_y - half_H_PX
        x2 = cur_x + half_W_PX
        y2 = cur_y + half_H_PX

        cv2.rectangle(canv_m, (x1, y1), (x2, y2), config.C_CAM, 2)
        cv2.circle(canv_m, (cur_x, cur_y), 3, config.C_MAP_ACT, -1)
    else:
        cv2.circle(canv_m, (cur_x, cur_y), 7, config.C_MAP_ACT, -1)

    txt_pos = f"X: {pos_m_x:.2f} cm, Y: {pos_m_y:.2f} cm"
    txt_esc = f"Escala: 1:{1.0/esc_m:.2f} px/cm"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.5, min(0.9, sz / 1000.0 + 0.4))
    thickness = 2
    padding = 8

    (tw1, th1), _ = cv2.getTextSize(txt_pos, font, font_scale, thickness)
    (tw2, th2), _ = cv2.getTextSize(txt_esc, font, font_scale, thickness)

    panel_w = max(tw1, tw2) + padding * 2
    panel_h = (th1 + th2) + padding * 3
    panel_x = 10
    panel_y = 10

    overlay = canv_m.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    alpha = 0.35
    canv_m[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w] = cv2.addWeighted(
        overlay[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w], alpha,
        canv_m[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w], 1 - alpha, 0
    )

    text_x = panel_x + padding
    text_y1 = panel_y + padding + th1
    text_y2 = text_y1 + padding + th2

    cv2.putText(canv_m, txt_pos, (text_x, text_y1), font, font_scale, config.C_MAP_TXT, thickness)
    cv2.putText(canv_m, txt_esc, (text_x, text_y2), font, font_scale, config.C_MAP_TXT, thickness)

    return cv2.resize(canv_m, (map_w_display, map_h_display))
