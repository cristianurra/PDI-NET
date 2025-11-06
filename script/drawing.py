import cv2
import numpy as np
import random
from config import ( # ¡CORREGIDO!
    MIN_DEPTH_CM, MAX_DEPTH_CM, C_MAP_FND, C_MAP_TXT, C_MAP_ACT, C_NARAN, C_GRIS, C_ACT, C_CAM,
    RAD_PUN, ESC_VEC, RECT_MARGIN_CM, MIN_SUPERVIVENCIA_FR, PORC_MOS, Q_X, Q_Y, Q_ACT_BASE
)
from utils import map_trans, depth_to_color # ¡CORREGIDO!

def dib_escala_profundidad(frame, w, h):
    BAR_W, BAR_H = 50, 300
    BAR_X = w - BAR_W - 20
    BAR_Y = max(10, h // 2 - BAR_H // 2)

    bar_img = np.zeros((BAR_H, BAR_W, 3), dtype=np.uint8)
    for i in range(BAR_H):
        norm_val = 1.0 - (i / BAR_H)
        depth_cm = MIN_DEPTH_CM + norm_val * (MAX_DEPTH_CM - MIN_DEPTH_CM)
        color = depth_to_color(depth_cm)
        bar_img[i, :, :] = color

    y_end = min(h, BAR_Y + BAR_H)
    x_end = min(w, BAR_X + BAR_W)
    if BAR_Y < y_end and BAR_X < x_end:
        frame[BAR_Y:y_end, BAR_X:x_end] = bar_img[0:(y_end - BAR_Y), 0:(x_end - BAR_X)]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_color = (255, 255, 255)

    text_cerca = f"{MIN_DEPTH_CM:.0f} cm"
    text_lejos = f"{MAX_DEPTH_CM:.0f} cm"
    label_cerca = "CERCA (ROJO)"
    label_lejos = "LEJOS (AZUL)"

    (tw_c, th_c), _ = cv2.getTextSize(text_cerca, font, font_scale, font_thickness)
    (tw_l, th_l), _ = cv2.getTextSize(text_lejos, font, font_scale, font_thickness)
    (tw_lbl_c, th_lbl_c), _ = cv2.getTextSize(label_cerca, font, font_scale, font_thickness)
    (tw_lbl_l, th_lbl_l), _ = cv2.getTextSize(label_lejos, font, font_scale, font_thickness)

    text_x = max(5, BAR_X - 10 - max(tw_c, tw_l, tw_lbl_c, tw_lbl_l))

    y_cerca = min(h - 5, max(th_c + 5, BAR_Y + 15))
    y_label_cerca = max(5 + th_lbl_c, BAR_Y - 15)
    y_lejos = min(h - 5, BAR_Y + BAR_H)
    y_label_lejos = min(h - 5, BAR_Y + BAR_H + 25)

    cv2.putText(frame, text_cerca, (text_x, y_cerca), font, font_scale, text_color, font_thickness)
    cv2.putText(frame, label_cerca, (text_x, max(5, y_label_cerca)), font, font_scale, text_color, font_thickness)
    cv2.putText(frame, text_lejos, (text_x, y_lejos), font, font_scale, text_color, font_thickness)
    cv2.putText(frame, label_lejos, (text_x, min(h - 5, y_label_lejos)), font, font_scale, text_color, font_thickness)

    cv2.line(frame, (max(0, BAR_X - 5), BAR_Y), (BAR_X, BAR_Y), text_color, 2)
    cv2.line(frame, (max(0, BAR_X - 5), BAR_Y + BAR_H), (BAR_X, BAR_Y + BAR_H), text_color, 2)


def dib_mov(frame, objs, w, h, depth_cm):

    vels_t = []

    objs_estables = [obj for obj in objs if obj['supervivencia_fr'] >= MIN_SUPERVIVENCIA_FR]
    num_mos = max(1, int(len(objs_estables) * PORC_MOS)) if objs_estables else 0
    objs_dib = random.sample(objs_estables, num_mos) if num_mos > 0 else []

    clean_area_image = frame[:, :w // 2].copy()

    for obj in objs_dib:
        cX, cY = obj['pos']
        cX_R, cY_R = obj['pos_R']
        color_punto = obj['color']

        cv2.circle(frame, (cX, cY), RAD_PUN, color_punto, -1)
        cv2.circle(frame, (cX_R, cY_R), RAD_PUN, color_punto, -1)

        v_x = np.median([v[0] for v in obj['hist_vel']])
        v_y = np.median([v[1] for v in obj['hist_vel']])

        if abs(v_x) > 1 or abs(v_y) > 1:
            vels_t.append((v_x, v_y))

    def dib_vec(vels, c_x, c_y, color, lbl=""):
        if vels:
            m_vx, m_vy = int(np.median([v[0] for v in vels])), int(np.median([v[1] for v in vels]))
            p2_x = c_x - m_vx * ESC_VEC
            p2_y = c_y - m_vy * ESC_VEC
            cv2.arrowedLine(frame, (c_x, c_y), (p2_x, p2_y), color, 4, tipLength=0.3)

            if lbl:
                cv2.putText(frame, lbl, (c_x - 40, c_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    v_cam_x, v_cam_y = 0, 0
    if vels_t:
        v_cam_x = int(np.median([v[0] for v in vels_t]))
        v_cam_y = int(np.median([v[1] for v in vels_t]))
        dib_vec(vels_t, w // 2, h // 4, C_CAM, 'CAMARA')

    txt_v_cam = f"MOV. HORIZ. (px): {v_cam_x}"
    cv2.putText(frame, txt_v_cam, (w // 2 - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_CAM, 2)

    if depth_cm > 0 and depth_cm < MAX_DEPTH_CM * 2:
        txt_depth = f"PROFUNDIDAD (cm): {depth_cm:.2f}"
    else:
        txt_depth = "PROFUNDIDAD: N/A o > 600 cm"

    cv2.putText(frame, txt_depth, (w // 2 - 150, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_CAM, 2)

    return -v_cam_x, -v_cam_y, clean_area_image


def dib_ayu(frame, w, h, q_w, q_h):
    m_x = w // 2
    cv2.line(frame, (m_x, 0), (m_x, h), C_NARAN, 2)
    cv2.putText(frame, 'OJO IZQ', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_NARAN, 2)
    cv2.putText(frame, 'OJO DER', (m_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_NARAN, 2)

    for col in range(1, Q_X):
        cv2.line(frame, (col * q_w, 0), (col * q_w, h), C_GRIS, 1)
    for row in range(1, Q_Y):
        cv2.line(frame, (0, row * q_h), (w, row * q_h), C_GRIS, 1)

    Q_ACT_DRAW = Q_ACT_BASE
    T_ACT = 3
    for row, col in Q_ACT_DRAW:
        x1, y1 = col * q_w, row * q_h
        x2, y2 = (col + 1) * q_w, (row + 1) * q_h
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_ACT, T_ACT)


def dib_map(hist_celdas_vis, pos_m_x, pos_m_y, fixed_grid_sz_cm, rect_sz_cm_actual, map_w_display, map_h_display, current_view_w_cm, current_view_h_cm):

    celdas_xy_cm = list(hist_celdas_vis.keys())

    if not celdas_xy_cm:
        celdas_xy_cm = [(pos_m_x, pos_m_y)]

    celdas_xy_cm.append((pos_m_x, pos_m_y))

    sz = max(map_w_display, map_h_display)
    canv_m = np.full((sz, sz, 3), C_MAP_FND, dtype=np.uint8)

    esc_m, off_x, off_y = map_trans(celdas_xy_cm, sz, sz)

    for (grid_x, grid_y), data in hist_celdas_vis.items():

        depth_c, image_c = data

        cell_center_x = grid_x * fixed_grid_sz_cm
        cell_center_y = grid_y * fixed_grid_sz_cm

        rect_size_cm = fixed_grid_sz_cm

        rect_size_px = int((rect_size_cm - RECT_MARGIN_CM) * esc_m)

        map_center_x = int(off_x + cell_center_x * esc_m)
        map_center_y = int(off_y - cell_center_y * esc_m)

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
    cur_y = int(off_y - pos_m_y * esc_m)

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

        cv2.rectangle(canv_m, (x1, y1), (x2, y2), C_CAM, 2)
        cv2.circle(canv_m, (cur_x, cur_y), 3, C_MAP_ACT, -1)
    else:
        cv2.circle(canv_m, (cur_x, cur_y), 7, C_MAP_ACT, -1)

    txt_pos = f"X: {pos_m_x:.2f} cm, Y: {pos_m_y:.2f} cm"
    txt_esc = f"Escala: 1:{1.0/esc_m:.2f} px/cm"

    # Textos del mapa con tamaño grande
    cv2.putText(canv_m, "MAPA DE ZONAS VISITADAS (CM) - IMAGEN POR VISTA", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, C_MAP_TXT, 3)
    cv2.putText(canv_m, txt_pos, (10, sz - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, C_MAP_TXT, 2)
    cv2.putText(canv_m, txt_esc, (sz - 360, sz - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, C_MAP_TXT, 2)

    return cv2.resize(canv_m, (map_w_display, map_h_display))

def show_compuesta(f_top, f_bottom_left_eye, canv_m, w_orig, h_orig):

    f_bottom_left_eye_bgr = cv2.cvtColor(f_bottom_left_eye, cv2.COLOR_GRAY2BGR)

    left_eye_w = f_bottom_left_eye_bgr.shape[1]
    map_display_w = w_orig - left_eye_w
    map_display_h = h_orig

    bottom_half_display = np.zeros((h_orig, w_orig, 3), dtype=np.uint8)

    bottom_half_display[0:h_orig, 0:left_eye_w] = f_bottom_left_eye_bgr

    map_offset_x = left_eye_w
    bottom_half_display[0:map_display_h, map_offset_x:map_offset_x + map_display_w] = canv_m

    full_display = cv2.vconcat([f_top, bottom_half_display])

    cv2.imshow('Interfaz Estéreo Unificada', full_display)
    return cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Interfaz Estéreo Unificada', cv2.WND_PROP_VISIBLE) < 1
