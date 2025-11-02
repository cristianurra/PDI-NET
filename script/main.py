import cv2
import numpy as np
from config import (
    NOM_VID, UMB_DIST, N_VEL_PR, Q_X, Q_Y, K_UNI, K_LIMP,
    FOCAL_PIX, BASELINE_CM, CM_POR_PX, FIXED_GRID_SIZE_CM, RECT_SZ_CM_FALLBACK
)
from tracker import Tracker
from stereo_processing import proc_seg, get_cns
from drawing import dib_escala_profundidad, dib_mov, dib_ayu, dib_map, show_compuesta
from utils import normalize_cell_view, register_image_to_map

def main():

    rect_sz_cm_actual = RECT_SZ_CM_FALLBACK

    cap = cv2.VideoCapture(NOM_VID)

    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir el video '{NOM_VID}'. Verifique la ruta del archivo.")
        return

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    q_w, q_h = w // Q_X, h // Q_Y

    tracker = Tracker(UMB_DIST, N_VEL_PR)

    pos_m_x, pos_m_y = 0.0, 0.0

    hist_celdas_vis = {}

    cv2.namedWindow('Interfaz Estéreo Unificada', cv2.WINDOW_NORMAL)

    ret, frame = cap.read()
    while ret:

        cns_filt = proc_seg(frame, K_UNI, K_LIMP)
        frame_top = frame.copy()

        cns_L_matched_only, matched_cns_pairs, cns_disp_only = get_cns(
            cns_filt, q_w, q_h, w
        )

        objs = tracker.update_and_get(matched_cns_pairs)

        depth_cm = 0.0
        if cns_disp_only:
            disparities = [d for c, d in cns_disp_only]
            disp_rep = np.median(disparities)
            if disp_rep > 1.0:
                depth_cm = (FOCAL_PIX * BASELINE_CM) / disp_rep

        dib_ayu(frame_top, w, h, q_w, q_h)
        del_p_x, del_p_y, vista_actual_limpia = dib_mov(frame_top, objs, w, h, depth_cm)
        dib_escala_profundidad(frame_top, w, h)

        del_c_x = del_p_x * CM_POR_PX
        del_c_y = del_p_y * CM_POR_PX

        pos_m_x += del_c_x
        pos_m_y += del_c_y

        current_view_w_cm = FIXED_GRID_SIZE_CM
        current_view_h_cm = FIXED_GRID_SIZE_CM

        if depth_cm > 0:
            ancho_total_cm_proyectado_ref = (depth_cm * (w / 2)) / FOCAL_PIX
            rect_sz_cm_actual = ancho_total_cm_proyectado_ref / Q_X
            rect_sz_cm_actual = np.clip(rect_sz_cm_actual, 10.0, 100.0)

            grid_x = round(pos_m_x / FIXED_GRID_SIZE_CM)
            grid_y = round(pos_m_y / FIXED_GRID_SIZE_CM)

            celda_id = (grid_x, grid_y)
            normalized_view = normalize_cell_view(vista_actual_limpia.copy(), cell_target_size=(100, 100))

            if celda_id not in hist_celdas_vis:
                hist_celdas_vis[celda_id] = (depth_cm, normalized_view)
            elif depth_cm < hist_celdas_vis[celda_id][0]:
                existing_image = hist_celdas_vis[celda_id][1]

                registered_image = register_image_to_map(normalized_view, existing_image)

                hist_celdas_vis[celda_id] = (depth_cm, registered_image)

        else:
            rect_sz_cm_actual = RECT_SZ_CM_FALLBACK

        cns_filt_left_eye = cns_filt[:, :w // 2]

        map_display_w = w - cns_filt_left_eye.shape[1]
        map_display_h = h

        canv_m = dib_map(
            hist_celdas_vis,
            pos_m_x,
            pos_m_y,
            FIXED_GRID_SIZE_CM,
            rect_sz_cm_actual,
            map_display_w,
            map_display_h,
            current_view_w_cm,
            current_view_h_cm
        )

        if show_compuesta(frame_top, cns_filt_left_eye, canv_m, w, h):
            break

        ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
