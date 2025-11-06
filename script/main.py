import os
import platform
import argparse #Para analizar argumentos de línea de comandos
import cv2
import numpy as np
from tqdm import tqdm
from config import (
    NOM_VID, UMB_DIST, N_VEL_PR, Q_X, Q_Y, K_UNI, K_LIMP,
    MESH_CONSOLIDATE_K, K_VERT_FILL, Y_MASK_OFFSET,
    FOCAL_PIX, BASELINE_CM, CM_POR_PX, FIXED_GRID_SIZE_CM, RECT_SZ_CM_FALLBACK,
    VAR_LAPLACIAN_THRESH, MAX_INITIAL_SKIP_FRAMES,
    EDGE_CANNY_LOW, EDGE_CANNY_HIGH, EDGE_MAX_POINTS, EDGE_POINT_RADIUS,
    ORANGE_HSV_LOW, ORANGE_HSV_HIGH, WHITE_INTENSITY_THRESH, WHITE_MORPH_K,
    EXPORT_ORANGE_CSV, ORANGE_CSV_PATH
)
from tracker import Tracker
from stereo_processing import proc_seg, get_cns, proc_mesh_mask, get_mesh_boundary_y_pos, get_mesh_boundary
from drawing import dib_escala_profundidad, dib_mov, dib_ayu, dib_map, show_compuesta
from utils import normalize_cell_view, register_image_to_map, detect_orange_markers

if platform.system() == 'Linux':
    # Sólo forzamos el backend Qt 'xcb' en Linux; en Windows no es necesario
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


def open_svo_file(svo_path, start_frame=0):
    """
    Abre un archivo .svo usando pyzed y retorna un generador de frames.
    
    Args:
        svo_path (str): Ruta al archivo .svo.
        start_frame (int): Frame desde donde iniciar la lectura.
    
    Returns:
        tuple: (generator, total_frames, width, height) o (None, 0, 0, 0) si falla.
    """
    try:
        import pyzed.sl as sl  # type: ignore
    except ImportError:
        print("ERROR: pyzed no está instalado. Para procesar archivos .svo, instala el ZED SDK y pyzed.")
        print("  Instalación: pip install pyzed")
        return None, 0, 0, 0
    
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR: No se pudo abrir el archivo SVO '{svo_path}': {status}")
        return None, 0, 0, 0
    
    # Obtener información del video (API 3.8)
    total_frames = zed.get_svo_number_of_frames()
    cam_info = zed.get_camera_information()
    w = cam_info.camera_resolution.width * 2  # Ancho total (izquierda + derecha)
    h = cam_info.camera_resolution.height
    
    # Posicionar en el frame inicial
    if start_frame > 0:
        if start_frame >= total_frames:
            print(f"ADVERTENCIA: Frame inicial {start_frame} excede el total de frames {total_frames}. Iniciando desde el frame 0.")
            start_frame = 0
        else:
            zed.set_svo_position(start_frame)
            print(f"Iniciando desde el frame {start_frame}/{total_frames}")
    
    print(f"Archivo SVO abierto: {total_frames} frames, resolución: {w}x{h}")
    
    def frame_generator():
        """Generador que extrae frames del archivo SVO."""
        left_image = sl.Mat()
        right_image = sl.Mat()
        
        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # Extraer imágenes izquierda y derecha
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                
                # Convertir a numpy arrays
                left_bgr = left_image.get_data()[:, :, :3]  # BGRA -> BGR
                right_bgr = right_image.get_data()[:, :, :3]
                
                # Concatenar horizontalmente (izquierda | derecha)
                stereo_frame = np.hstack((left_bgr, right_bgr))
                
                yield True, stereo_frame
            else:
                yield False, None
                zed.close()
                break
    
    return frame_generator(), total_frames, w, h


def main():
    ap = argparse.ArgumentParser(description="Proyecto de Procesamiento de Imágenes Estéreo")
    ap.add_argument("-v", "--video", type=str, default=NOM_VID,
                    help="Ruta al archivo de video de entrada (por defecto: valor de NOM_VID en config.py)")
    ap.add_argument("-sf", "--start-frame", type=int, default=0,
                    help="Frame inicial desde donde comenzar el procesamiento (por defecto: 0)")
    args = ap.parse_args()
    
    rect_sz_cm_actual = RECT_SZ_CM_FALLBACK
    
    # Detectar tipo de archivo
    ext = os.path.splitext(args.video)[1].lower()
    is_svo = ext == '.svo'
    
    # Variables comunes
    cap = None
    frame_generator = None
    w, h, total_frames = 0, 0, 0
    start_frame = args.start_frame
    
    if is_svo:
        # Procesar archivo .svo
        frame_generator, total_frames, w, h = open_svo_file(args.video, start_frame)
        if frame_generator is None:
            return
    else:
        # Procesar video estándar (MP4, AVI, etc.)
        cap = cv2.VideoCapture(args.video)

        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            cap = cv2.VideoCapture(args.video, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            print(f"ERROR: No se pudo abrir el video '{args.video}'. Asegúrate que la ruta es correcta y que OpenCV soporta el códec.")
            print("Sugerencia: coloca un MP4 accesible y actualiza `NOM_VID` en `script\\config.py` con la ruta absoluta, por ejemplo:\n  NOM_VID = r\"C:\\ruta\\a\\mi_video.mp4\"\n")
            return

        # Obtener dimensiones del video
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Posicionar en el frame inicial para MP4
        if start_frame > 0:
            if start_frame >= total_frames:
                print(f"ADVERTENCIA: Frame inicial {start_frame} excede el total de frames {total_frames}. Iniciando desde el frame 0.")
                start_frame = 0
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                print(f"Iniciando desde el frame {start_frame}/{total_frames}")
    # Calcular dimensiones de la cuadrícula
    q_w, q_h = w // Q_X, h // Q_Y

    # Inicializar el tracker
    tracker = Tracker(UMB_DIST, N_VEL_PR)

    pos_m_x, pos_m_y = 0.0, 0.0

    hist_celdas_vis = {}

    cv2.namedWindow('Interfaz Estéreo Unificada', cv2.WINDOW_NORMAL) # el nombre debe soportar tildes
    cv2.resizeWindow('Interfaz Estéreo Unificada', 1280, 720)

    # Pre-scan: saltar frames iniciales muy borrosos usando la varianza del Laplaciano
    skipped_initial = 0
    first_good_frame = None
    
    # No me funciona bien el laplaciano, por lo que lo comento por mientras, luego lo mejoramos
    # De momento puse la opcion de empezar de fotogramas especificos
    # if is_svo:
    #     # Para SVO, leer directamente desde el generador
    #     for i in range(int(MAX_INITIAL_SKIP_FRAMES)):
    #         ok, ftmp = next(frame_generator)
    #         if not ok:
    #             break
    #         try:
    #             gray_tmp = cv2.cvtColor(ftmp, cv2.COLOR_BGR2GRAY)
    #             lap_var = cv2.Laplacian(gray_tmp, cv2.CV_64F).var()
    #         except Exception:
    #             lap_var = 0.0

    #         if lap_var >= VAR_LAPLACIAN_THRESH:
    #             first_good_frame = ftmp
    #             break
    #         skipped_initial += 1
    # else:
    #     # Para MP4, usar VideoCapture
    #     for i in range(int(MAX_INITIAL_SKIP_FRAMES)):
    #         ok, ftmp = cap.read()
    #         if not ok:
    #             break
    #         try:
    #             gray_tmp = cv2.cvtColor(ftmp, cv2.COLOR_BGR2GRAY)
    #             lap_var = cv2.Laplacian(gray_tmp, cv2.CV_64F).var()
    #         except Exception:
    #             lap_var = 0.0

    #         if lap_var >= VAR_LAPLACIAN_THRESH:
    #             first_good_frame = ftmp
    #             break
    #         skipped_initial += 1

    # if first_good_frame is not None:
    #     print(f"Se saltaron {skipped_initial} frames iniciales borrosos (lap_var={lap_var:.1f}); comenzando desde un frame nítido.")
    #     frame = first_good_frame
    #     ret = True
    #     frame_idx = skipped_initial
    # else:
    #     # No se encontró frame suficientemente nítido: volver al inicio y usar el primer frame disponible
    #     if is_svo:
    #         # Para SVO, necesitamos recrear el generador (no se puede rebobinar)
    #         frame_generator, _, _, _ = open_svo_file(args.video)
    #         ret, frame = next(frame_generator)
    #     else:
    #         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #         ret, frame = cap.read()
    #     if ret:
    #         print("No se detectó frame suficientemente nítido al inicio; usando primer frame disponible.")
    #     frame_idx = 0
    
    # Leer el primer frame desde la posición inicial
    if is_svo:
        ret, frame = next(frame_generator)
    else:
        ret, frame = cap.read()
    frame_idx = start_frame

    # CSV export disabled by default; not creating any CSV unless explicitly enabled in config

    # Configurar barra de progreso: inicia desde start_frame y va hasta total_frames
    pbar = tqdm(total=total_frames, initial=start_frame, desc="Procesando frames", unit="frame")
    # main loop
    while ret:

        cns_filt = proc_seg(frame, K_UNI, K_LIMP)

        # -------------------------------------------------------------
        # Detección del Borde (solo para posición Y de enmascaramiento)
        # -------------------------------------------------------------
        # 1. Generar máscara consolidada usando el canal de saturación
        mesh_mask = proc_mesh_mask(frame, MESH_CONSOLIDATE_K, K_LIMP, K_VERT_FILL)

        # 2. Encontrar la coordenada Y más alta del borde
        y_borde_detectado = get_mesh_boundary_y_pos(mesh_mask, w // 2, h, K_LIMP)

        # 3. Definir el límite máximo de Y para el tracking (aplicando offset)
        # Si la detección del borde falla (devuelve 0), usamos el FALLBACK seguro y_max_track = 0
        # (acepta todos los puntos). Si se detecta el borde, aplicamos el offset para definir
        # el inicio del tracking más abajo.
        if y_borde_detectado > 0 and y_borde_detectado < h:
            # Si el borde fue detectado (y_borde_detectado > 0), el tracking empieza un poco más abajo (offset).
            y_max_track = y_borde_detectado + Y_MASK_OFFSET
        else:
            # FALLBACK SEGURO: Si la detección falla, el límite superior de tracking es Y=0
            # (es decir, acepta todos los puntos, ya que cY > 0 es casi siempre True).
            y_max_track = 0

        frame_top = frame.copy()

        # OJO: Se pasa el nuevo límite 'y_max_track' a get_cns
        cns_L_matched_only, matched_cns_pairs, cns_disp_only = get_cns(
            cns_filt, q_w, q_h, w, y_max_track=y_max_track
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

        # -------------------------------------------------------------
        # 1. Consolidación de la Malla (Apertura + Cierre Vertical)
        # -------------------------------------------------------------
        # Usamos MESH_CONSOLIDATE_K y K_VERT_FILL para fusionar las líneas de la malla en una región sólida.
        mesh_mask = proc_mesh_mask(frame, MESH_CONSOLIDATE_K, K_LIMP, K_VERT_FILL)

        # 2. Detección de Borde de Malla Superior (Borde Morfológico: A - A $\ominus$ B)
        # K_LIMP es el kernel de 3x3 para obtener el borde fino del objeto consolidado.
        mesh_boundary_contour = get_mesh_boundary(
            mesh_mask, w // 2, h, K_LIMP
        )

        # Dibujar el contorno del borde de la malla (solo en el ojo izquierdo)
        if mesh_boundary_contour is not None:
            C_MESH_EDGE = (255, 255, 0) # Color Cian
            cv2.drawContours(frame_top, [mesh_boundary_contour], -1, C_MESH_EDGE, 2)
            cv2.putText(frame_top, 'BORDE MALLA', (10, h // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_MESH_EDGE, 2)

        # Dibujado del límite de tracking y del borde detectado para diagnóstico
        # y_max_track > 0 significa que la detección fue exitosa y aplicamos el offset.
        if y_max_track > 0:
            C_MASK_LINE = (0, 0, 255) # Límite de Tracking (Rojo)
            C_DETECTION = (0, 255, 255) # Borde Detectado Puro (Amarillo)

            # Línea ROJA (LÍMITE FINAL DE TRACKING)
            cv2.line(frame_top, (0, y_max_track), (w, y_max_track), C_MASK_LINE, 2)
            cv2.putText(frame_top, 'LIMITE TRACKING', (10, max(10, y_max_track - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_MASK_LINE, 2)

            # LÍNEA AMARILLA (BORDE PURO MEDIANO DETECTADO, sin offset)
            if y_borde_detectado > 0:
                cv2.line(frame_top, (0, y_borde_detectado), (w, y_borde_detectado), C_DETECTION, 1)
                cv2.putText(frame_top, 'BORDE DETECTADO', (10, max(10, y_borde_detectado - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_DETECTION, 1)


        # --- Detección y Dibujo de Lazos Naranjas (Marcadores de Referencia) ---
        try:
            # Iterar sobre el ojo izquierdo (0) y el ojo derecho (w//2)
            for x_start in [0, w // 2]:
                x_end = x_start + (w // 2)
                eye = frame[:, x_start:x_end].copy() 
                markers = detect_orange_markers(eye)
                
                for m in markers:
                    gx = x_start + m['cx'] 
                    gy = m['cy'] 
                    
                    C_NARANJA_DRAW = (0, 140, 255) # Naranja oscuro
                    cv2.circle(frame_top, (gx, gy), EDGE_POINT_RADIUS + 4, C_NARANJA_DRAW, -1)
                    cv2.putText(frame_top, 'REF', (gx + 5, gy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_NARANJA_DRAW, 2)

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
        if is_svo:
            ret, frame = next(frame_generator)
        else:
            ret, frame = cap.read()
        frame_idx += 1
        pbar.update(1) # Actualiza la barra de progreso

        # safety: if frame index didn't advance or read failed repeatedly, break
        if not ret and frame_idx - prev_frame <= 1:
            break
    
    
    # Cierre Programa 
    pbar.close()
    if cap is not None:
        cap.release()
    # no CSV to close (export disabled)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
