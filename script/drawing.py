 
import cv2
import numpy as np
import random
from config import *
from utils import map_trans

def dib_mov(frame, objs, w, h):

    m_x = w // 2
    vels_i, vels_d, vels_t = [], [], []

    objs_estables = [obj for obj in objs if obj['supervivencia_fr'] >= MIN_SUPERVIVENCIA_FR]
    num_mos = max(1, int(len(objs_estables) * PORC_MOS)) if objs_estables else 0
    objs_dib = random.sample(objs_estables, num_mos) if num_mos > 0 else []

    for obj in objs_dib:
        cX, cY = obj['pos']

        color_punto = obj['color'] if obj['supervivencia_fr'] >= MIN_SUPERVIVENCIA_FR else C_GRIS
        cv2.circle(frame, (cX, cY), RAD_PUN, color_punto, -1)

        v_x, v_y = np.mean([v[0] for v in obj['hist_vel']]), np.mean([v[1] for v in obj['hist_vel']])

        if abs(v_x) > 1 or abs(v_y) > 1:
            vels_t.append((v_x, v_y))
            # Separar por ojo (izquierda o derecha)
            (vels_i if cX < m_x else vels_d).append((v_x, v_y))

    def dib_vec(vels, c_x, c_y, color, lbl=""):
        if vels:
            m_vx, m_vy = int(np.median([v[0] for v in vels])), int(np.median([v[1] for v in vels]))
            p2_x = c_x - m_vx * ESC_VEC
            p2_y = c_y - m_vy * ESC_VEC
            cv2.arrowedLine(frame, (c_x, c_y), (p2_x, p2_y), color, 4, tipLength=0.3)

            if lbl:
                cv2.putText(frame, lbl, (c_x - 40, c_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    dib_vec(vels_i, w // 4, h // 2, C_VEC_LR)
    dib_vec(vels_d, w * 3 // 4, h // 2, C_VEC_LR)

    v_cam_x, v_cam_y = 0, 0
    if vels_t:
        v_cam_x = int(np.median([v[0] for v in vels_t]))
        v_cam_y = int(np.median([v[1] for v in vels_t]))
        dib_vec([(v_cam_x, v_cam_y)], w // 2, h // 4, C_CAM, 'CAMARA')

    return -v_cam_x, -v_cam_y

def dib_ayu(frame, w, h, q_w, q_h):
    m_x = w // 2
    cv2.line(frame, (m_x, 0), (m_x, h), C_NARAN, 2)
    cv2.putText(frame, 'OJO IZQ', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_NARAN, 2)
    cv2.putText(frame, 'OJO DER', (m_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_NARAN, 2)
    for col in range(1, Q_X):
        cv2.line(frame, (col * q_w, 0), (col * q_w, h), C_GRIS, 1)
    for row in range(1, Q_Y):
        cv2.line(frame, (0, row * q_h), (w, row * q_h), C_GRIS, 1)

    T_ACT = 3
    for row, col in Q_ACT:
        x1, y1 = col * q_w, row * q_h
        x2, y2 = (col + 1) * q_w, (row + 1) * q_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_ACT, T_ACT)

def dib_map(hist_m, pos_m_x, pos_m_y):
    sz = MAP_CAN_SZ
    canv_m = np.full((sz, sz, 3), C_MAP_FND, dtype=np.uint8)
    esc_m, off_x, off_y = map_trans(hist_m, sz, sz)
    pts_m = []
    for x_c, y_c in hist_m:
        x_map = int(off_x + x_c * esc_m)
        y_map = int(off_y - y_c * esc_m)
        pts_m.append((x_map, y_map))
    for i in range(1, len(pts_m)):
        cv2.line(canv_m, pts_m[i-1], pts_m[i], C_MAP_TRA, 2)

    cur_x = int(off_x + pos_m_x * esc_m)
    cur_y = int(off_y - pos_m_y * esc_m)
    cv2.circle(canv_m, (cur_x, cur_y), 5, C_MAP_ACT, -1)

    txt_pos = f"X: {pos_m_x:.2f} cm, Y: {pos_m_y:.2f} cm"
    txt_esc = f"Escala: 1:{1.0/esc_m:.2f} px/cm"

    cv2.putText(canv_m, "MAPA DE RECORRIDO (CM)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_MAP_TXT, 1)
    cv2.putText(canv_m, txt_pos, (10, sz - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_MAP_TXT, 1)
    cv2.putText(canv_m, txt_esc, (sz - 150, sz - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_MAP_TXT, 1)

    return canv_m

def show_v(f_p, canv_m, n_w, n_h):
    f_red = cv2.resize(f_p, (n_w, n_h))
    cv2.imshow('Video Principal', f_red)
    m_red = cv2.resize(canv_m, (MAP_DISP_SZ, MAP_DISP_SZ))
    cv2.imshow('Mapa de Recorrido', m_red)
    return cv2.waitKey(1) & 0xFF == ord('q')
