import os
import platform
import cv2
import numpy as np
from config import (
    NOM_VID, UMB_DIST, N_VEL_PR, Q_X, Q_Y, K_UNI, K_LIMP,
    FOCAL_PIX, BASELINE_CM, CM_POR_PX, FIXED_GRID_SIZE_CM, RECT_SZ_CM_FALLBACK,
    VAR_LAPLACIAN_THRESH, MAX_INITIAL_SKIP_FRAMES,
    EDGE_CANNY_LOW, EDGE_CANNY_HIGH, EDGE_MAX_POINTS, EDGE_POINT_RADIUS,
    ORANGE_HSV_LOW, ORANGE_HSV_HIGH, WHITE_INTENSITY_THRESH, WHITE_MORPH_K,
    EXPORT_ORANGE_CSV, ORANGE_CSV_PATH
)
from tracker import Tracker
from stereo_processing import proc_seg, get_cns
from drawing import dib_escala_profundidad, dib_mov, dib_ayu, dib_map, show_compuesta
from utils import normalize_cell_view, register_image_to_map, detect_orange_markers

if platform.system() == 'Linux':
    # Sólo forzamos el backend Qt 'xcb' en Linux; en Windows no es necesario
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

def main():

    rect_sz_cm_actual = RECT_SZ_CM_FALLBACK

    cap = cv2.VideoCapture(NOM_VID)

    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        cap = cv2.VideoCapture(NOM_VID, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        ext = os.path.splitext(NOM_VID)[1].lower()
        if ext == '.svo':
            print(f"ERROR: No se pudo abrir el archivo SVO '{NOM_VID}'. OpenCV no soporta archivos .svo directamente.")
            print('  - Opción rápida: abre el .svo con ZED Explorer y exporta a MP4 (File -> Export)')
            print('  - Opción programática: usa ZED SDK / pyzed para leer el SVO y guardarlo como MP4 o secuencia de frames.')
        else:
            print(f"ERROR: No se pudo abrir el video '{NOM_VID}'. Asegúrate que la ruta es correcta y que OpenCV soporta el códec.")

        print("Sugerencia: coloca un MP4 accesible y actualiza `NOM_VID` en `script\\config.py` con la ruta absoluta, por ejemplo:\n  NOM_VID = r\"C:\\ruta\\a\\mi_video.mp4\"\n")
        return

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    q_w, q_h = w // Q_X, h // Q_Y

    tracker = Tracker(UMB_DIST, N_VEL_PR)

    pos_m_x, pos_m_y = 0.0, 0.0

    hist_celdas_vis = {}

    cv2.namedWindow('Interfaz Estéreo Unificada', cv2.WINDOW_NORMAL)

    # Pre-scan: saltar frames iniciales muy borrosos usando la varianza del Laplaciano
    skipped_initial = 0
    first_good_frame = None
    for i in range(int(MAX_INITIAL_SKIP_FRAMES)):
        ok, ftmp = cap.read()
        if not ok:
            break
        try:
            gray_tmp = cv2.cvtColor(ftmp, cv2.COLOR_BGR2GRAY)
            lap_var = cv2.Laplacian(gray_tmp, cv2.CV_64F).var()
        except Exception:
            lap_var = 0.0

        if lap_var >= VAR_LAPLACIAN_THRESH:
            first_good_frame = ftmp
            break
        skipped_initial += 1

    if first_good_frame is not None:
        print(f"Se saltaron {skipped_initial} frames iniciales borrosos (lap_var={lap_var:.1f}); comenzando desde un frame nítido.")
        frame = first_good_frame
        ret = True
        frame_idx = skipped_initial
    else:
        # No se encontró frame suficientemente nítido: volver al inicio y usar el primer frame disponible
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            print("No se detectó frame suficientemente nítido al inicio; usando primer frame disponible.")
        frame_idx = 0

    # CSV export disabled by default; not creating any CSV unless explicitly enabled in config

    # main loop
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

        # Detectar solo puntos naranjas en ambos ojos y marcarlos (no borrar nada)
        try:
            for eye_i, x_start in enumerate([0, w // 2]):
                x_end = x_start + (w // 2)
                eye = frame[:, x_start:x_end]
                markers = detect_orange_markers(eye)
                for m in markers:
                    gx = x_start + m['cx']
                    gy = m['cy']
                    cv2.circle(frame_top, (gx, gy), EDGE_POINT_RADIUS + 4, (0, 140, 255), -1)
                    cv2.putText(frame_top, 'REF', (gx + 5, gy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,140,255), 2)

                    # no CSV export (disabled)
        except Exception:
            pass

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

        # avanzar al siguiente frame
        prev_frame = frame_idx
        ret, frame = cap.read()
        frame_idx += 1

        # safety: if frame index didn't advance or read failed repeatedly, break
        if not ret and frame_idx - prev_frame <= 1:
            break

    cap.release()
    # no CSV to close (export disabled)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
