import os
import cv2
import numpy as np
import math
import time
import threading
import json
import csv
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from typing import List, Tuple, Dict, Any

from config import ConfiguracionGlobal
from mapper import GlobalMapper2D
from tracker import Tracker
from utils import open_svo_file, normalize_cell_view, register_image_to_map
from stereo_processing import proc_seg, proc_mesh_mask, get_mesh_boundary_y_pos, get_cns, detect_orange_markers
from drawing import dib_ayu, dib_mov, dib_escala_profundidad, dib_map
from anomaly_detector import DamageDetector
from yolo_tracker import YOLOTracker, YOLOOverlayDrawer
from visual_odometry import VisualOdometry, AdaptiveTrajectoryDrawer
from hardware_optimizer import initialize_hardware_optimization, get_cuda_processor


class ProcesadorEstereoThread(threading.Thread):
    def __init__(self, config: ConfiguracionGlobal, gui_ref, cuda_processor=None):
        super().__init__()
        self.config = config
        self.gui_ref = gui_ref
        self._running = True
        self._stop_event = threading.Event()  # Para cierre más limpio
        self.is_paused = False
        self.mapeo = GlobalMapper2D(config)
        self.hist_celdas_vis: Dict[Tuple[int, int], Tuple[float, np.ndarray]] = {}
        self.tracked_objects_history: List[List[Dict[str, Any]]] = []
        self.damage_detector = DamageDetector(config)
        self.damage_log: List[Dict[str, Any]] = [] 
        self.last_map_radar = None
        self.yolo_tracker = YOLOTracker(config)
        self.visual_odometry = VisualOdometry(config)
        self.odometry_drawer = AdaptiveTrajectoryDrawer(canvas_width=400, canvas_height=300)
        self.cuda_processor = cuda_processor
        
        # Tracking de posición por supervivencia
        self.pos_supervivencia_x = 0.0
        self.pos_supervivencia_y = 0.0
        self.trajectory_supervivencia = []
        
        # Matrices 4x4 para exportación Open3D
        self.matrices_yolo = []
        self.matrices_supervivencia = []
        
        # Tracking de progreso del video
        self.current_frame = 0
        self.total_frames = 0
        
        # Marcadores de detección YOLO (bordes, nudos)
        self.yolo_markers = []  # Lista de {'frame_index', 'class', 'name', 'id', 'marker_id', 'pos_x', 'pos_y'}
        self.marcador_counter = 0  # Contador para marcadores (clase 0)
        self.nudo_counter = 0  # Contador para nudos (clase 1)
        self.marker_proximity_threshold = 20.0  # cm - distancia mínima para considerar duplicado

        self.marcadores_contados = 0
        self.distancia_real_acumulada = 0.0
        self.distancia_puntos_actual = 0.0
        self.distancia_yolo_actual = 0.0
        self.mejor_algoritmo = "Recopilando datos..."
        self.color_ganador = "#FFFFFF"
        
        # Variables para medir distancia entre marcadores consecutivos
        self.ultimo_marcador_pos_yolo = None  # (x, y) en el último marcador
        self.ultimo_marcador_pos_superv = None  # (x, y) en el último marcador

        self.ids_marcadores_procesados = set()
        
        # Lista para guardar mediciones de marcadores en CSV
        self.mediciones_marcadores = []
        
        # Variables para FPS
        self.fps_current = 0.0
        self.fps_average = 0.0
        self.fps_frame_times = []
        self.fps_last_time = time.time()
        
        # Carpeta única para esta sesión
        self.session_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.session_folder = os.path.join('report', f'sesion_{self.session_timestamp}')
        os.makedirs(self.session_folder, exist_ok=True)
        
        # Metadatos de la sesión
        self.session_start_time = datetime.now()
        self.frames_procesados = 0

    def stop(self):
        print("[STOP] Solicitando detención del thread...")
        self._running = False
        self._stop_event.set()  # Señalar evento de parada
        # Guardar datos de tracking al detener
        try:
            self._save_tracking_data()
        except Exception as e:
            print(f" Error al guardar datos: {e}")
    
    def _save_tracking_data(self):
        """Guarda los datos de tracking en archivos JSON, CSV e imágenes organizados por sesión."""
        try:
            import json
            import platform
            
            session_end_time = datetime.now()
            duracion_sesion = (session_end_time - self.session_start_time).total_seconds()
            
            print(f"\n{'='*60}")
            print(f"GUARDANDO DATOS DE SESIÓN: {self.session_folder}")
            print(f"{'='*60}")
            print(f"  - matrices_yolo: {len(self.matrices_yolo)} frames")
            print(f"  - matrices_supervivencia: {len(self.matrices_supervivencia)} frames")
            print(f"  - mediciones_marcadores: {len(self.mediciones_marcadores)} registros")
            print(f"  - imágenes capturadas: {len(self.hist_celdas_vis)} celdas")
            
            # 1. GUARDAR INFORMACIÓN DE LA SESIÓN (TXT)
            info_path = os.path.join(self.session_folder, 'info_sesion.txt')
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("INFORMACIÓN DE LA SESIÓN DE PROCESAMIENTO\n")
                f.write("="*60 + "\n\n")
                
                # Información del video
                f.write("--- INFORMACIÓN DEL VIDEO ---\n")
                f.write(f"Archivo: {os.path.basename(self.config.NOM_VID)}\n")
                f.write(f"Ruta completa: {self.config.NOM_VID}\n")
                f.write(f"Formato: {os.path.splitext(self.config.NOM_VID)[1].upper()}\n")
                f.write(f"Fotogramas totales: {self.total_frames}\n")
                f.write(f"Fotogramas procesados: {self.frames_procesados}\n")
                f.write(f"Frame inicial: {self.config.START_FRAME}\n")
                f.write(f"Skip rate: {self.config.SKIP_RATE}\n\n")
                
                # Información del sistema
                f.write("--- INFORMACIÓN DEL EQUIPO ---\n")
                f.write(f"Sistema operativo: {platform.system()} {platform.release()}\n")
                f.write(f"Procesador: {platform.processor()}\n")
                f.write(f"Arquitectura: {platform.machine()}\n")
                f.write(f"Python: {platform.python_version()}\n")
                try:
                    import cv2
                    f.write(f"OpenCV: {cv2.__version__}\n")
                except:
                    pass
                f.write("\n")
                
                # Información de la sesión
                f.write("--- INFORMACIÓN DE LA SESIÓN ---\n")
                f.write(f"Inicio: {self.session_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Fin: {session_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duración: {duracion_sesion:.2f} segundos ({duracion_sesion/60:.2f} minutos)\n")
                f.write(f"FPS promedio: {self.fps_average:.2f}\n\n")
                
                # Configuración de tracking
                f.write("--- CONFIGURACIÓN DE TRACKING ---\n")
                f.write(f"Sistema de capturas: {self.config.SISTEMA_CAPTURAS_AUTO}\n")
                f.write(f"YOLO tracking: {'Activado' if self.config.YOLO_TRACKING_ENABLED else 'Desactivado'}\n")
                f.write(f"Profundidad estéreo: {'Activado' if self.config.PROFUNDIDAD_ESTEREO else 'Desactivado'}\n")
                f.write(f"Vector supervivencia: {'Visible' if self.config.MOSTRAR_VECTOR_SUPERVIVENCIA else 'Oculto'}\n")
                f.write(f"Vector YOLO: {'Visible' if self.config.MOSTRAR_VECTOR_YOLO else 'Oculto'}\n\n")
                
                # Estadísticas de odometría
                f.write("--- ESTADÍSTICAS DE ODOMETRÍA ---\n")
                yolo_pos = self.visual_odometry.get_position()
                f.write(f"Posición final YOLO: ({yolo_pos[0]:.2f}, {yolo_pos[1]:.2f}) cm\n")
                f.write(f"Posición final Supervivencia: ({self.pos_supervivencia_x:.2f}, {self.pos_supervivencia_y:.2f}) cm\n")
                f.write(f"Trayectoria YOLO: {len(self.visual_odometry.get_trajectory())} puntos\n")
                f.write(f"Trayectoria Supervivencia: {len(self.trajectory_supervivencia)} puntos\n\n")
                
                # Estadísticas de marcadores
                f.write("--- ESTADÍSTICAS DE MARCADORES ---\n")
                f.write(f"Marcadores detectados (Borde): {self.marcador_counter}\n")
                f.write(f"Nudos detectados: {self.nudo_counter}\n")
                f.write(f"Total marcadores YOLO: {len(self.yolo_markers)}\n")
                f.write(f"Mediciones de distancia: {len(self.mediciones_marcadores)}\n")
                if self.mediciones_marcadores:
                    f.write(f"Algoritmo ganador: {self.mejor_algoritmo}\n")
                f.write("\n")
                
                # Estadísticas de captura
                f.write("--- ESTADÍSTICAS DE CAPTURA ---\n")
                f.write(f"Celdas capturadas: {len(self.hist_celdas_vis)}\n")
                f.write(f"Tamaño de grilla: 50.0 cm\n")
                if self.hist_celdas_vis:
                    depths = [depth for (depth, _) in self.hist_celdas_vis.values()]
                    f.write(f"Profundidad promedio: {sum(depths)/len(depths):.2f} cm\n")
                    f.write(f"Profundidad mínima: {min(depths):.2f} cm\n")
                    f.write(f"Profundidad máxima: {max(depths):.2f} cm\n")
                f.write("\n")
                
                f.write("="*60 + "\n")
                f.write("FIN DEL REPORTE\n")
                f.write("="*60 + "\n")
            
            print(f"✓ Información de sesión guardada en {info_path}")
            
            # 2. GUARDAR MATRICES DE ODOMETRÍA (JSON)
            if self.matrices_yolo:
                json_yolo_path = os.path.join(self.session_folder, 'odometria_yolo.json')
                with open(json_yolo_path, 'w') as f:
                    json.dump(self.matrices_yolo, f, indent=2)
                print(f"✓ Guardados {len(self.matrices_yolo)} frames YOLO en {json_yolo_path}")
            
            if self.matrices_supervivencia:
                json_superv_path = os.path.join(self.session_folder, 'odometria_supervivencia.json')
                with open(json_superv_path, 'w') as f:
                    json.dump(self.matrices_supervivencia, f, indent=2)
                print(f"✓ Guardados {len(self.matrices_supervivencia)} frames Supervivencia en {json_superv_path}")
            
            # 3. GUARDAR MEDICIONES DE MARCADORES (CSV)
            if self.mediciones_marcadores:
                csv_path = os.path.join(self.session_folder, 'mediciones_marcadores.csv')
                with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['timestamp', 'marcador_desde', 'marcador_hasta', 
                                'distancia_real_cm', 'distancia_yolo_cm', 'error_yolo_cm',
                                'distancia_supervivencia_cm', 'error_supervivencia_cm',
                                'mejor_algoritmo', 'sistema_capturas_usado']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.mediciones_marcadores)
                print(f"✓ Guardadas {len(self.mediciones_marcadores)} mediciones en {csv_path}")
            
            # 4. GUARDAR IMÁGENES CAPTURADAS
            if self.hist_celdas_vis:
                images_dir = os.path.join(self.session_folder, 'capturas')
                os.makedirs(images_dir, exist_ok=True)
                
                for (grid_x, grid_y), (depth, image) in self.hist_celdas_vis.items():
                    img_filename = f'celda_x{grid_x}_y{grid_y}_depth{depth:.1f}cm.png'
                    img_path = os.path.join(images_dir, img_filename)
                    cv2.imwrite(img_path, image)
                
                print(f"✓ Guardadas {len(self.hist_celdas_vis)} imágenes en {images_dir}")
            
            print(f"\n{'='*60}")
            print(f"✅ TODOS LOS DATOS GUARDADOS EN: {self.session_folder}")
            print(f"{'='*60}\n")
                
        except Exception as e:
            import traceback
            print(f" Error guardando tracking data: {e}")
            traceback.print_exc()

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def run(self):
        # Limpiar datos de tracking anteriores al inicio de nueva simulación
        self.matrices_yolo = []
        self.matrices_supervivencia = []
        self.pos_supervivencia_x = 0.0
        self.pos_supervivencia_y = 0.0
        self.trajectory_supervivencia = []
        self.yolo_markers = []  # Limpiar marcadores
        self.marcador_counter = 0
        self.nudo_counter = 0
        self.ultimo_marcador_pos_yolo = None
        self.ultimo_marcador_pos_superv = None
        self.ids_marcadores_procesados = set()
        self.marcadores_contados = 0
        self.distancia_real_acumulada = 0.0
        
        # Limpiar archivos JSON existentes
        import json
        try:
            with open(self.config.OUTPUT_JSON_YOLO, 'w') as f:
                json.dump([], f)
            with open(self.config.OUTPUT_JSON_SUPERVIVENCIA, 'w') as f:
                json.dump([], f)
            print("✓ Archivos JSON inicializados (vacíos)")
        except Exception as e:
            print(f" Error al limpiar archivos JSON: {e}")
        
        # Inyectar procesador CUDA en stereo_processing
        if self.cuda_processor:
            import stereo_processing
            stereo_processing.set_cuda_processor(self.cuda_processor)
        
        ext = os.path.splitext(self.config.NOM_VID)[1].lower()
        is_svo = ext == '.svo'
        frame_generator = None
        cap = None

        if is_svo:
            frame_generator, total_frames, w, h = open_svo_file(self.config.NOM_VID)
            self.total_frames = total_frames
            if frame_generator is None or w == 0:
                self.gui_ref.root.after(0, lambda: messagebox.showerror("Error", "Error al abrir."))
                return
        else:
            cap = cv2.VideoCapture(self.config.NOM_VID)
            if not cap.isOpened():
                self.gui_ref.root.after(0, lambda: messagebox.showerror("Error", f"No se pudo abrir el video: {self.config.NOM_VID}"))
                return
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.START_FRAME)

        if w == 0 or h == 0:
            self.gui_ref.root.after(0, lambda: messagebox.showerror("Error", "El video tiene dimensiones inválidas."))
            return

        q_w, q_h = w // self.config.Q_X, h // self.config.Q_Y
        
        # Función auxiliar para dibujar texto con sombra mejorada
        def draw_text_with_shadow(img, text, pos, font, scale, color, thickness):
            x, y = pos
            # Sombra negra más gruesa y con offset mayor para mejor contraste
            cv2.putText(img, text, (x+3, y+3), font, scale, (0, 0, 0), thickness+2, cv2.LINE_AA)
            # Texto blanco principal
            cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

        tracker = Tracker(self.config.UMB_DIST, self.config.N_VEL_PR, self.config)
        pos_m_x, pos_m_y = 0.0, 0.0

        ret, frame = (next(frame_generator) if is_svo else cap.read())
        frame_counter = 0

        while self._running and ret:

            while self.is_paused and self._running:
                time.sleep(0.1)

            if not self._running:
                break
            
            # Calcular FPS
            current_time = time.time()
            frame_time = current_time - self.fps_last_time
            self.fps_last_time = current_time
            
            if frame_time > 0:
                self.fps_current = 1.0 / frame_time
                self.fps_frame_times.append(frame_time)
                # Mantener solo los últimos 30 frames para el promedio
                if len(self.fps_frame_times) > 30:
                    self.fps_frame_times.pop(0)
                # Calcular FPS promedio
                avg_frame_time = sum(self.fps_frame_times) / len(self.fps_frame_times)
                self.fps_average = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

            if frame_counter % self.config.SKIP_RATE == 0:

                self.config.K_UNI = np.ones((self.config.K_UNI_SIZE, self.config.K_UNI_SIZE), np.uint8)
                self.config.K_LIMP = np.ones((self.config.K_LIMP_SIZE, self.config.K_LIMP_SIZE), np.uint8)
                self.config.K_VERT_FILL = cv2.getStructuringElement(cv2.MORPH_RECT, (self.config.K_VERT_FILL_W, self.config.K_VERT_FILL_H))

                tracker.update_config(self.config.UMB_DIST, self.config.N_VEL_PR)

                cns_filt = proc_seg(frame, self.config.K_UNI, self.config.K_LIMP)
                mesh_mask = proc_mesh_mask(frame, self.config.MESH_CONSOLIDATE_K, self.config.K_LIMP, self.config.K_VERT_FILL)
                y_borde_detectado = get_mesh_boundary_y_pos(mesh_mask, w // 2, h, self.config.K_LIMP)

                y_max_track = h

                frame_top = frame.copy()

                cns_L_matched_only, matched_cns_pairs, cns_disp_only = get_cns(
                    cns_filt, q_w, q_h, w, self.config, y_max_track=y_max_track
                )

                objs = tracker.update_and_get(matched_cns_pairs)

                self.tracked_objects_history.append(objs)
                if len(self.tracked_objects_history) > self.config.N_FRAMES_HISTORIAL:
                    self.tracked_objects_history.pop(0)

                valid_depth_count = sum(1 for o in objs if o.get('depth_cm', 0) > 0)
                quality_good = valid_depth_count > 3

                if quality_good:
                    self.mapeo.update_position(objs)

                depth_cm = 0.0
                if self.config.PROFUNDIDAD_STEREO_ACTIVA and cns_disp_only:
                    disparities = [d for c, d in cns_disp_only]
                    disp_rep = np.median(disparities)
                    if disp_rep > 1.0:
                        depth_cm = (self.config.FOCAL_PIX * self.config.BASELINE_CM) / disp_rep

                frame_left = frame[:, :w//2]
                
                # IDs de marcadores duplicados (compartido entre detección y medición)
                ids_duplicados_ignorados = set()
                
                # Tracking YOLO en vista izquierda ANTES de damage detector
                if self.config.YOLO_TRACKING_ENABLED:
                    frame_left_for_yolo = frame_left.copy()
                    frame_tracked, vectors_x, vectors_y, detections = self.yolo_tracker.track_frame(frame_left_for_yolo)
                    
                    # Actualizar odometría visual
                    self.visual_odometry.update(vectors_x, vectors_y)
                    
                    # Guardar matrices YOLO inmediatamente después de actualizar odometría
                    yolo_pos = self.visual_odometry.get_position()
                    mat_yolo = np.eye(4)
                    mat_yolo[0, 3] = yolo_pos[0] / 100.0
                    mat_yolo[1, 3] = -yolo_pos[1] / 100.0
                    mat_yolo[2, 3] = 0.0
                    self.matrices_yolo.append(mat_yolo.tolist())
                    for det in detections:
                        if det['crossed_center']:  # Ambas clases: 0=marcador, 1=Nudo
                            # Usar la longitud actual de la trayectoria como índice
                            traj_len = len(self.visual_odometry.get_trajectory())
                            mat_len = len(self.matrices_yolo)
                            
                            # Verificar si ya existe un marcador cercano del mismo tipo
                            is_duplicate = False
                            for existing_marker in self.yolo_markers:
                                if existing_marker['class'] == det['class']:
                                    # Calcular distancia desde la posición del marcador existente
                                    existing_pos = self.visual_odometry.get_trajectory()[existing_marker['frame_index']]
                                    current_pos = yolo_pos
                                    dist = ((existing_pos[0] - current_pos[0])**2 + 
                                           (existing_pos[1] - current_pos[1])**2)**0.5
                                    
                                    if dist < self.marker_proximity_threshold:
                                        is_duplicate = True
                                        ids_duplicados_ignorados.add(det['id'])  # Marcar como duplicado
                                        print(f"Marcador duplicado ignorado: {det['name']} a {dist:.1f}cm del marcador {existing_marker['marker_id']}")
                                        break
                            
                            if not is_duplicate:
                                # Incrementar contador según clase
                                if det['class'] == 0:  # Marcador (Borde)
                                    self.marcador_counter += 1
                                    marker_id = f"M{self.marcador_counter}"
                                else:  # Nudo
                                    self.nudo_counter += 1
                                    marker_id = f"N{self.nudo_counter}"
                                
                                marker = {
                                    'frame_index': traj_len - 1,  # Índice de la trayectoria actual
                                    'class': det['class'],
                                    'name': det['name'],
                                    'id': det['id'],
                                    'marker_id': marker_id,
                                    'pos_x': yolo_pos[0],
                                    'pos_y': yolo_pos[1]
                                }
                                self.yolo_markers.append(marker)
                                print(f"Marcador {marker['marker_id']}: {det['name']} (ID:{det['id']}) en frame_idx={marker['frame_index']}, pos=({yolo_pos[0]:.1f}, {yolo_pos[1]:.1f}) cm [traj_len={traj_len}, mat_len={mat_len}]")
                    
                    # Usar frame con tracking como base
                    frame_left = frame_tracked
                else:
                    vectors_x, vectors_y, detections = [], [], []
                
                
                # Detectar daños sobre frame con tracking
                frame_with_damages, damages_info = self.damage_detector.detect(frame_left)  
                frame_top[:, :w//2] = frame_with_damages

                for dmg in damages_info:
                    dmg['frame'] = frame_counter
                    dmg['global_x'] = pos_m_x
                    dmg['global_y'] = pos_m_y
                    self.damage_log.append(dmg)
                
                self.distancia_puntos_actual = math.hypot(self.pos_supervivencia_x, self.pos_supervivencia_y)
                #yolo_pos = self.visual_odometry.get_position()
                self.distancia_yolo_actual = math.hypot(yolo_pos[0], yolo_pos[1]) if self.config.YOLO_TRACKING_ENABLED else 0.0

                for det in detections:
                    if det['class'] == 0 and det['crossed_center']:
                        marker_id = det['id']
                        # Ignorar si fue marcado como duplicado o ya procesado
                        if marker_id in ids_duplicados_ignorados:
                            continue
                        if marker_id not in self.ids_marcadores_procesados:
                            self.marcadores_contados += 1
                            self.ids_marcadores_procesados.add(marker_id)
                            
                            # Obtener posiciones actuales
                            pos_yolo_actual = yolo_pos if self.config.YOLO_TRACKING_ENABLED else (0.0, 0.0)
                            pos_superv_actual = (self.pos_supervivencia_x, self.pos_supervivencia_y)
                            
                            # Si es el PRIMER marcador, solo guardamos la posición de referencia
                            if self.marcadores_contados == 1:
                                self.ultimo_marcador_pos_yolo = pos_yolo_actual
                                self.ultimo_marcador_pos_superv = pos_superv_actual
                                print(f"\nMarcador {self.marcadores_contados} (INICIO): Posiciones guardadas como referencia")
                                print(f"   YOLO: ({pos_yolo_actual[0]:.1f}, {pos_yolo_actual[1]:.1f}) cm")
                                print(f"   Superv: ({pos_superv_actual[0]:.1f}, {pos_superv_actual[1]:.1f}) cm")
                            else:
                                # A partir del SEGUNDO marcador, calculamos distancia desde el anterior
                                distancia_real = 100.0  # 1 metro entre marcadores
                                
                                # Calcular distancia recorrida por YOLO desde último marcador
                                if self.ultimo_marcador_pos_yolo:
                                    delta_yolo = math.hypot(
                                        pos_yolo_actual[0] - self.ultimo_marcador_pos_yolo[0],
                                        pos_yolo_actual[1] - self.ultimo_marcador_pos_yolo[1]
                                    )
                                else:
                                    delta_yolo = 0.0
                                
                                # Calcular distancia recorrida por Supervivencia desde último marcador
                                if self.ultimo_marcador_pos_superv:
                                    delta_superv = math.hypot(
                                        pos_superv_actual[0] - self.ultimo_marcador_pos_superv[0],
                                        pos_superv_actual[1] - self.ultimo_marcador_pos_superv[1]
                                    )
                                else:
                                    delta_superv = 0.0
                                
                                # Calcular errores
                                error_yolo = abs(distancia_real - delta_yolo)
                                error_superv = abs(distancia_real - delta_superv)
                                
                                # Determinar cuál es mejor
                                if error_yolo < error_superv:
                                    self.mejor_algoritmo = "YOLO"
                                    self.color_ganador = "#00FF00"
                                elif error_superv < error_yolo:
                                    self.mejor_algoritmo = "Supervivencia"
                                    self.color_ganador = "#FFFF00"
                                else:
                                    self.mejor_algoritmo = "Empate"
                                    self.color_ganador = "#FFFFFF"
                                
                                # Elegir sistema usado según configuración
                                if self.config.SISTEMA_CAPTURAS_AUTO == 'supervivencia':
                                    sistema_usado = "Supervivencia"
                                else:
                                    sistema_usado = "YOLO"
                                
                                # Mostrar resultados
                                print(f"\nMarcador {self.marcadores_contados-1} -> {self.marcadores_contados}:")
                                print(f"   Real:         {distancia_real:.1f} cm")
                                print(f"   YOLO:         {delta_yolo:.1f} cm (error: {error_yolo:.1f} cm)")
                                print(f"   Supervivencia: {delta_superv:.1f} cm (error: {error_superv:.1f} cm)")
                                print(f"   Mejor: {self.mejor_algoritmo} | Sistema capturas: {sistema_usado}")
                                
                                # Guardar medición en la lista para CSV
                                medicion = {
                                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                    'marcador_desde': self.marcadores_contados - 1,
                                    'marcador_hasta': self.marcadores_contados,
                                    'distancia_real_cm': round(distancia_real, 2),
                                    'distancia_yolo_cm': round(delta_yolo, 2),
                                    'error_yolo_cm': round(error_yolo, 2),
                                    'distancia_supervivencia_cm': round(delta_superv, 2),
                                    'error_supervivencia_cm': round(error_superv, 2),
                                    'mejor_algoritmo': self.mejor_algoritmo,
                                    'sistema_capturas_usado': sistema_usado
                                }
                                self.mediciones_marcadores.append(medicion)
                                
                                # Actualizar posiciones de referencia para el próximo marcador
                                self.ultimo_marcador_pos_yolo = pos_yolo_actual
                                self.ultimo_marcador_pos_superv = pos_superv_actual

                dib_ayu(frame_top, w, h, q_w, q_h, self.config)
                
                # IMPORTANTE: Capturar imagen cruda ANTES de dibujar vectores
                # Usar el frame original sin procesar del lado izquierdo
                frame_left_raw = frame[:, :w//2].copy()
                
                del_p_x, del_p_y, vista_actual_limpia = dib_mov(frame_top, objs, w, h, depth_cm, self.config, self.config.MOSTRAR_VECTOR_SUPERVIVENCIA)
                dib_escala_profundidad(frame_top, w, h, self.config)
                
                # Dibujar vector YOLO si está habilitado (en vista izquierda siempre)
                if self.config.MOSTRAR_VECTOR_YOLO:
                    from drawing import dib_vector_yolo
                    # Aplicar en la porción izquierda del frame
                    if self.config.VISTA_MONO:
                        dib_vector_yolo(frame_top, w//2, h, 
                                       self.visual_odometry.velocity_x, 
                                       self.visual_odometry.velocity_y, 
                                       self.config)
                    else:
                        # En modo stereo, dibujar solo en la mitad izquierda
                        left_frame = frame_top[:, :w//2]
                        dib_vector_yolo(left_frame, w//2, h, 
                                       self.visual_odometry.velocity_x, 
                                       self.visual_odometry.velocity_y, 
                                       self.config)

                
                
                try:
                    for x_start in [0, w // 2]:
                        x_end = x_start + (w // 2)
                        eye = frame[:, x_start:x_end].copy()
                        markers = detect_orange_markers(eye, self.config)

                        for m in markers:
                            gx = x_start + m['cx']
                            gy = m['cy']
                            C_NARANJA_DRAW = (0, 140, 255)
                            cv2.circle(frame_top, (gx, gy), self.config.EDGE_POINT_RADIUS + 4, C_NARANJA_DRAW, -1)
                except Exception:
                    pass


                del_c_x = del_p_x * self.config.CM_POR_PX
                del_c_y = del_p_y * self.config.CM_POR_PX
                pos_m_x += del_c_x
                pos_m_y += del_c_y
                
                # Actualizar posición de supervivencia para odometría
                # Invertir Y porque en imagen Y+ es abajo, pero en coordenadas Y+ es arriba
                self.pos_supervivencia_x += del_c_x
                self.pos_supervivencia_y -= del_c_y  # Invertir Y
                
                # Debug: Mostrar valores cada 30 frames
                if frame_counter % 30 == 0:
                    print(f"Frame {frame_counter}: del_p_x={del_p_x:.2f}, del_p_y={del_p_y:.2f}, "
                          f"pos_superv=({self.pos_supervivencia_x:.2f}, {self.pos_supervivencia_y:.2f})")
                
                # Guardar matrices 4x4 SIEMPRE (mismo formato para ambas trayectorias)
                # Convertir centímetros a metros dividiendo por 100
                # Y invertir para que coincida con convención de Open3D
                
                # Matriz de supervivencia - SIEMPRE guardar aunque del_p sea 0
                mat_superv = np.eye(4)
                mat_superv[0, 3] = self.pos_supervivencia_x / 100.0  # cm a m
                mat_superv[1, 3] = -self.pos_supervivencia_y / 100.0  # Invertir Y
                mat_superv[2, 3] = 0.0
                self.matrices_supervivencia.append(mat_superv.tolist())
                
                # Matriz YOLO (usando la misma posición en cm)
                yolo_pos = self.visual_odometry.get_position()
                mat_yolo = np.eye(4)
                mat_yolo[0, 3] = yolo_pos[0] / 100.0  # cm a m
                mat_yolo[1, 3] = -yolo_pos[1] / 100.0  # Invertir Y
                mat_yolo[2, 3] = 0.0
                self.matrices_yolo.append(mat_yolo.tolist())

                rect_sz_cm_actual = self.config.RECT_SZ_CM_FALLBACK
                if depth_cm > 0:
                    ancho_total_cm_proyectado_ref = (depth_cm * (w / 2)) / self.config.FOCAL_PIX
                    rect_sz_cm_actual = rect_sz_cm_actual / self.config.Q_X
                    rect_sz_cm_actual = np.clip(rect_sz_cm_actual, 10.0, 100.0)

                    # Determinar qué posición usar según configuración
                    if self.config.SISTEMA_CAPTURAS_AUTO == 'yolo':
                        capture_pos_x = yolo_pos[0]
                        capture_pos_y = yolo_pos[1]
                    else:
                        capture_pos_x = pos_m_x
                        capture_pos_y = pos_m_y
                    
                    # Usar grid más pequeño para capturas más frecuentes (50cm en lugar de FIXED_GRID_SIZE_CM)
                    capture_grid_size = 50.0  # cm
                    grid_x = round(capture_pos_x / capture_grid_size)
                    grid_y = round(capture_pos_y / capture_grid_size)
                    celda_id = (grid_x, grid_y)
                    
                    # Capturar solo si la celda está vacía (intentando no dejar huecos)
                    if celda_id not in self.hist_celdas_vis:
                        # Guardar imagen cruda completa de alta calidad (sin normalizar)
                        self.hist_celdas_vis[celda_id] = (depth_cm, frame_left_raw.copy())
                        sistema_usado = "YOLO" if self.config.SISTEMA_CAPTURAS_AUTO == 'yolo' else "Supervivencia"
                        print(f"Captura en celda ({grid_x}, {grid_y}) con {sistema_usado}: pos=({capture_pos_x:.1f}, {capture_pos_y:.1f}) cm")

                cns_filt_left_eye = cns_filt[:, :w // 2]
                map_display_w = 400
                map_display_h = 400

                map_radar = self.mapeo.draw_map(objs, frames_history=self.tracked_objects_history)

                self.last_map_radar = map_radar

                # Determinar qué posición mostrar en el mapa según sistema de capturas
                if self.config.SISTEMA_CAPTURAS_AUTO == 'yolo':
                    display_pos_x = yolo_pos[0]
                    display_pos_y = yolo_pos[1]
                else:
                    display_pos_x = pos_m_x
                    display_pos_y = pos_m_y
                
                canv_m = dib_map(
                    self.hist_celdas_vis, display_pos_x, display_pos_y, self.config.FIXED_GRID_SIZE_CM,
                    rect_sz_cm_actual, map_display_w, map_display_h,
                    self.config.FIXED_GRID_SIZE_CM, self.config.FIXED_GRID_SIZE_CM, self.config
                )

                # Calcular velocidad de supervivencia para el gráfico de odometría
                del_c_x_frame = del_p_x * self.config.CM_POR_PX if 'del_p_x' in locals() else 0.0
                del_c_y_frame = del_p_y * self.config.CM_POR_PX if 'del_p_y' in locals() else 0.0

                # Generar gráfico de odometría visual
                odometry_graph = self.odometry_drawer.draw(
                    self.visual_odometry.get_trajectory(),
                    self.visual_odometry.get_position(),
                    (self.visual_odometry.velocity_x, self.visual_odometry.velocity_y),
                    self.visual_odometry.status,
                    self.visual_odometry.status_color,
                    trajectory2=self.trajectory_supervivencia,
                    current_pos2=(self.pos_supervivencia_x, self.pos_supervivencia_y),
                    velocity2=(del_c_x_frame, del_c_y_frame),
                    markers=self.yolo_markers  # Pasar marcadores YOLO
                )
                
                # Dibujar FPS en la esquina superior izquierda (ANTES de enviar a GUI)
                fps_text_current = f"FPS: {self.fps_current:.1f}"
                fps_text_avg = f"Avg: {self.fps_average:.1f}"
                # Usar tamaño de fuente más grande (0.8) y grosor 2 para mejor visibilidad
                draw_text_with_shadow(frame_top, fps_text_current, (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                draw_text_with_shadow(frame_top, fps_text_avg, (10, 65), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                self.gui_ref.root.after(0, self.gui_ref.actualizar_gui, frame_top, cns_filt_left_eye, canv_m, map_radar, odometry_graph, depth_cm, display_pos_x, display_pos_y, self.mapeo.global_angle, self.current_frame, self.total_frames)
            
            # IMPORTANTE: Guardar matrices en CADA frame para trazado 3D completo
            # (Fuera del if SKIP_RATE para garantizar continuidad)
            
            # Actualizar posición de supervivencia (ya calculado arriba para odometría)
            # Invertir Y porque en imagen Y+ es abajo, pero en coordenadas Y+ es arriba
            self.pos_supervivencia_x += del_c_x_frame
            self.pos_supervivencia_y -= del_c_y_frame  # Invertir Y
            
            # Matriz de supervivencia - guardar en cada frame
            mat_superv = np.eye(4)
            mat_superv[0, 3] = self.pos_supervivencia_x / 100.0
            mat_superv[1, 3] = -self.pos_supervivencia_y / 100.0
            mat_superv[2, 3] = 0.0
            self.matrices_supervivencia.append(mat_superv.tolist())
            
            # Guardar trayectoria de supervivencia para el 2D
            self.trajectory_supervivencia.append((self.pos_supervivencia_x, self.pos_supervivencia_y))
            
            # Si YOLO no está habilitado, guardar matriz YOLO con valores por defecto
            if not self.config.YOLO_TRACKING_ENABLED:
                yolo_pos = self.visual_odometry.get_position()
                mat_yolo = np.eye(4)
                mat_yolo[0, 3] = yolo_pos[0] / 100.0
                mat_yolo[1, 3] = -yolo_pos[1] / 100.0
                mat_yolo[2, 3] = 0.0
                self.matrices_yolo.append(mat_yolo.tolist())

            frame_counter += 1
            self.current_frame = self.config.START_FRAME + frame_counter
            self.frames_procesados = frame_counter  # Actualizar contador de frames procesados

            # Verificar si debemos detenernos antes de sleep
            if not self._running or self._stop_event.is_set():
                print("[THREAD] Deteniendo antes de sleep...")
                break
                
            time.sleep(0.001)

            # Verificar nuevamente antes de leer siguiente frame
            if not self._running or self._stop_event.is_set():
                print("[THREAD] Deteniendo antes de leer frame...")
                break
                
            ret, frame = (next(frame_generator) if is_svo else cap.read())

        # Limpieza al terminar
        print("Limpiando recursos del thread...")
        if cap:
            cap.release()
        if is_svo and frame_generator:
            # Cerrar generador SVO si existe
            try:
                frame_generator.close()
            except:
                pass
        
        # Guardar datos finales
        self._save_tracking_data()
        
        print("✓ Thread de procesamiento finalizado")
        self.gui_ref.root.after(0, lambda: self.gui_ref.depth_label.config(text="Procesamiento TERMINADO."))


class StereoAppTkinter:
    VIDEO_WIDTH_FIXED = 1440
    VIDEO_HEIGHT_FIXED = 480
    LAST_FILE_CONFIG = "last_video_config.json"

    def __init__(self, root: tk.Tk, config: ConfiguracionGlobal):
        self.root = root
        self.config = config
        self.root.title("Sistema de Procesamiento Estéreo (Tkinter)")
        
        # Inicializar optimizaciones de hardware
        self.hw_optimizer, self.cuda_processor = initialize_hardware_optimization()

        try:
            # Intentar maximizar, pero si falla, usar tamaño HD por defecto
            self.root.geometry("1280x720+50+50")  # HD (720p) centrado
            self.root.state('zoomed')
        except tk.TclError:
            # Si no puede maximizar, mantener tamaño HD
            self.root.geometry("1280x720+50+50")
            self.root.state('normal')

        self.thread: ProcesadorEstereoThread = None
        self.tk_image_video = None
        self.tk_image_mask = None
        self.tk_image_map = None
        self.tk_image_radar = None
        self.tk_image_odometry = None

        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.var_profundidad_stereo = tk.BooleanVar(value=self.config.PROFUNDIDAD_STEREO_ACTIVA)
        self.var_vista_mono = tk.BooleanVar(value=self.config.VISTA_MONO)
        self.var_mostrar_vector_supervivencia = tk.BooleanVar(value=self.config.MOSTRAR_VECTOR_SUPERVIVENCIA)
        self.var_mostrar_vector_yolo = tk.BooleanVar(value=self.config.MOSTRAR_VECTOR_YOLO)
        self.var_capturas_yolo = tk.BooleanVar(value=(self.config.SISTEMA_CAPTURAS_AUTO == 'yolo'))

        # Configurar cierre correcto de la aplicación
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self._select_file_and_start()
    
    def _on_closing(self):
        """Maneja el cierre correcto de la aplicación."""
        print("\n" + "="*60)
        print("Cerrando aplicación...")
        print("="*60)
        
        # Detener el thread de procesamiento si está activo
        if self.thread and self.thread.is_alive():
            print("Deteniendo thread de procesamiento...")
            self.thread.stop()
            # No esperar - es daemon, se cerrará automáticamente
        
        # Cerrar todas las ventanas de OpenCV si existen
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        print("✓ Cerrando aplicación")
        print("="*60 + "\n")
        
        # Destruir ventana de tkinter y salir
        try:
            self.root.quit()
        except:
            pass
        
        try:
            self.root.destroy()
        except:
            pass
        
        # Salida inmediata
        import os
        os._exit(0)  # Salida forzada inmediata (más rápida que sys.exit)

    def _load_last_config(self):
        """Carga la configuración del último archivo usado."""
        try:
            if os.path.exists(self.LAST_FILE_CONFIG):
                with open(self.LAST_FILE_CONFIG, 'r') as f:
                    data = json.load(f)
                    return data.get('file_path'), data.get('start_frame', 500)
        except Exception as e:
            print(f"Error cargando última configuración: {e}")
        return None, None
    
    def _save_last_config(self, file_path: str, start_frame: int):
        """Guarda la configuración del último archivo usado."""
        try:
            with open(self.LAST_FILE_CONFIG, 'w') as f:
                json.dump({'file_path': file_path, 'start_frame': start_frame}, f)
        except Exception as e:
            print(f"Error guardando última configuración: {e}")


    def _select_file_and_start(self):
        self.root.withdraw()
        
        # Intentar cargar último archivo usado
        last_file, last_frame = self._load_last_config()
        
        file_path = None
        start_frame = 500
        
        # Si existe último archivo, preguntar si quiere usarlo
        if last_file and os.path.exists(last_file):
            use_last = messagebox.askyesno(
                "Último archivo usado",
                f"¿Desea abrir el último video usado?\n\nArchivo: {os.path.basename(last_file)}\nFrame inicial: {last_frame}",
                parent=self.root
            )
            if use_last:
                file_path = last_file
                start_frame = last_frame
        
        # Si no hay último archivo o el usuario rechazó, pedir nuevo archivo
        if not file_path:
            file_path = filedialog.askopenfilename(
                title="1. Seleccionar Video Estéreo (MP4 o SVO)",
                filetypes=[("Archivos de Video", "*.mp4 *.avi *.svo"), ("Todos los archivos", "*.*")],
                initialdir=os.path.dirname(last_file) if last_file else None
            )

            if not file_path:
                self.root.quit()
                return

            while True:
                try:
                    start_frame_str = simpledialog.askstring("2. Frame Inicial",
                                                             "Ingrese el frame desde el que desea iniciar el procesamiento (Ej: 0, 500):",
                                                             initialvalue=str(last_frame) if last_frame else "500",
                                                             parent=self.root)
                    if start_frame_str is None:
                        self.root.quit()
                        return

                    start_frame = int(start_frame_str)
                    if start_frame >= 0:
                        break
                    else:
                        messagebox.showerror("Error", "El frame inicial debe ser un número positivo.")
                except ValueError:
                    messagebox.showerror("Error", "Entrada inválida. Debe ser un número entero.")
        
        # Guardar configuración para la próxima vez
        self._save_last_config(file_path, start_frame)


        self.config.NOM_VID = file_path
        self.config.START_FRAME = start_frame
        self._setup_gui()
        self.root.deiconify()
        self.start_processing_thread()

    def _setup_gui(self):

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.place(relx=0, rely=0, relwidth=1, relheight=1)

        video_col_frame = ttk.Frame(main_frame)
        video_col_frame.place(relx=0, rely=0, relwidth=0.75, relheight=1)
        video_col_frame.grid_columnconfigure(0, weight=1)
        video_col_frame.grid_rowconfigure(0, weight=0)
        video_col_frame.grid_rowconfigure(1, weight=1)
        video_col_frame.grid_rowconfigure(2, weight=0)

        video_container = ttk.Frame(video_col_frame, width=self.VIDEO_WIDTH_FIXED, height=self.VIDEO_HEIGHT_FIXED)
        video_container.grid(row=0, column=0, sticky="nw", padx=5, pady=5)
        video_container.grid_propagate(False)

        self.video_label = ttk.Label(video_container, text=f"Cargando: {os.path.basename(self.config.NOM_VID)}", anchor="center", background="#333", foreground="#fff")
        self.video_label.pack(fill="both", expand=True)

        # Frame para mapa radar y odometría visual (dividido horizontalmente)
        radar_odometry_frame = ttk.Frame(video_col_frame)
        radar_odometry_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        radar_odometry_frame.grid_columnconfigure(0, weight=1)
        radar_odometry_frame.grid_columnconfigure(1, weight=1)
        radar_odometry_frame.grid_rowconfigure(0, weight=1)
        
        # Mapa 3D Radar (mitad izquierda, tamaño reducido)
        self.mapa_radar_label = ttk.Label(radar_odometry_frame, text="Mapa 3D (Radar)", background="#000", foreground="#fff")
        self.mapa_radar_label.grid(row=0, column=0, sticky="nsew", padx=(0, 2))
        
        # Gráfico de Odometría Visual (mitad derecha)
        self.odometry_label = ttk.Label(radar_odometry_frame, text="Odometría Visual", background="#000", foreground="#fff")
        self.odometry_label.grid(row=0, column=1, sticky="nsew", padx=(2, 0))

        info_control_frame = ttk.Frame(video_col_frame)
        info_control_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=2)
        self.depth_label = ttk.Label(info_control_frame, text="Profundidad: N/A")
        self.pos_label = ttk.Label(info_control_frame, text="Posición Global: X: 0.0, Y: 0.0")
        self.angle_label = ttk.Label(info_control_frame, text="Ángulo: 0.0°")

        self.depth_label.pack(side="left", padx=5)
        self.pos_label.pack(side="left", padx=5)
        self.angle_label.pack(side="left", padx=5)

        ttk.Button(info_control_frame, text="⏸️ Pausar", command=self.pause_thread).pack(side="right", padx=5)
        self.play_button = ttk.Button(info_control_frame, text="▶️ Reanudar", command=self.resume_thread, state=tk.DISABLED)
        self.play_button.pack(side="right", padx=5)
        ttk.Button(info_control_frame, text="🗺️ Mapa 3D", command=self.show_3d_map).pack(side="right", padx=5)
        
        # Línea de tiempo del video
        timeline_frame = ttk.Frame(video_col_frame)
        timeline_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        self.timeline_label = ttk.Label(timeline_frame, text="Frame: 0 / 0")
        self.timeline_label.pack(side="left", padx=5)
        
        self.progress_bar = ttk.Progressbar(timeline_frame, mode='determinate', length=400)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=5)

        control_col_frame = ttk.Frame(main_frame)
        control_col_frame.place(relx=0.75, rely=0, relwidth=0.25, relheight=1)
        control_col_frame.grid_columnconfigure(0, weight=1)
        control_col_frame.grid_rowconfigure(0, weight=1)
        control_col_frame.grid_rowconfigure(1, weight=1)
        control_col_frame.grid_rowconfigure(2, weight=0)
        control_col_frame.grid_rowconfigure(3, weight=0)

        self.mask_label = ttk.Label(control_col_frame, text="Máscara Binaria", background="#333", foreground="#fff")
        self.mask_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.mapa_posicion_label = ttk.Label(control_col_frame, text="Mapa de Zonas 2D", background="#000", foreground="#fff")
        self.mapa_posicion_label.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self._create_control_panel(control_col_frame).grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        ttk.Button(control_col_frame, text=" Terminar Ejecución", command=self.terminar_ejecucion, style='Danger.TButton').grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        # Botón para cambiar el video
        ttk.Button(control_col_frame, text="🔄 Cambiar Video", command=self.change_video).grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        ttk.Button(control_col_frame, text="💾 Guardar Reporte", command=self.guardar_reporte).grid(row=5, column=0, sticky="ew", padx=5, pady=5)

        self.style.configure('Danger.TButton', foreground='red', font=('Helvetica', 10, 'bold'))

    def _create_control_panel(self, parent):
        group = ttk.LabelFrame(parent, text="Ajuste de Parámetros de Tracking", padding="10")

        switch_frame = ttk.Frame(group)
        switch_frame.pack(fill='x', pady=5)

        ttk.Label(switch_frame, text="Profundidad Estéreo:").pack(side="left", padx=(10, 0))
        ttk.Checkbutton(switch_frame, variable=self.var_profundidad_stereo, command=self._update_switches, style='TCheckbutton').pack(side="left", padx=5)
        
        ttk.Label(switch_frame, text="Vista Mono:").pack(side="left", padx=(20, 0))
        ttk.Checkbutton(switch_frame, variable=self.var_vista_mono, command=self._update_switches, style='TCheckbutton').pack(side="left", padx=5)
        
        # Segunda fila de switches
        switch_frame2 = ttk.Frame(group)
        switch_frame2.pack(fill='x', pady=5)
        
        ttk.Label(switch_frame2, text="Vector Supervivencia:").pack(side="left", padx=(10, 0))
        ttk.Checkbutton(switch_frame2, variable=self.var_mostrar_vector_supervivencia, command=self._update_switches, style='TCheckbutton').pack(side="left", padx=5)
        
        ttk.Label(switch_frame2, text="Vector YOLO:").pack(side="left", padx=(20, 0))
        ttk.Checkbutton(switch_frame2, variable=self.var_mostrar_vector_yolo, command=self._update_switches, style='TCheckbutton').pack(side="left", padx=5)
        
        # Tercera fila: Sistema de capturas automáticas
        switch_frame3 = ttk.Frame(group)
        switch_frame3.pack(fill='x', pady=5)
        
        ttk.Label(switch_frame3, text="Capturas con YOLO:").pack(side="left", padx=(10, 0))
        ttk.Checkbutton(switch_frame3, variable=self.var_capturas_yolo, command=self._update_switches, style='TCheckbutton').pack(side="left", padx=5)
        ttk.Label(switch_frame3, text="(desmarcar para usar Supervivencia)", foreground="gray").pack(side="left", padx=(5, 0))

        self._add_slider(group, "Distancia Umbral (UMB_DIST)", 'UMB_DIST', 10, 200, 5)
        self._add_slider(group, "Min. Supervivencia (FR)", 'MIN_SUPERVIVENCIA_FR', 1, 60, 1)
        self._add_slider(group, "Tolerancia Y Estéreo", 'Y_TOLERANCE', 1, 20, 1)
        self._add_slider(group, "Disparidad Mínima (px)", 'MIN_DISPARITY', 1, 50, 1)
        self._add_slider(group, "Muestreo de Puntos (%)", 'PORC_MOS_INT', 1, 100, 1)
        self._add_slider(group, "Skip Rate (1/N Frames)", 'SKIP_RATE', 1, 15, 1)
        self._add_slider(group, "Historial Mapa 3D (Frames)", 'N_FRAMES_HISTORIAL', 1, 30, 1)
        self._add_input(group, "Profundidad Mínima (cm)", 'MIN_DEPTH_CM', 1)
        self._add_input(group, "Profundidad Máxima (cm)", 'MAX_DEPTH_CM', 1)

        self._add_slider(group, "Kernel Unión (K_UNI)", 'K_UNI_SIZE', 1, 15, 2)
        self._add_slider(group, "Kernel Limpieza (K_LIMP)", 'K_LIMP_SIZE', 1, 9, 2)

        return group

    def _update_switches(self):
        self.config.PROFUNDIDAD_STEREO_ACTIVA = self.var_profundidad_stereo.get()
        self.config.VISTA_MONO = self.var_vista_mono.get()
        self.config.MOSTRAR_VECTOR_SUPERVIVENCIA = self.var_mostrar_vector_supervivencia.get()
        self.config.MOSTRAR_VECTOR_YOLO = self.var_mostrar_vector_yolo.get()
        self.config.SISTEMA_CAPTURAS_AUTO = 'yolo' if self.var_capturas_yolo.get() else 'supervivencia'

    def _add_slider(self, parent, text, param_key, min_val, max_val, step):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=2)
        current_value = getattr(self.config, param_key)
        label = ttk.Label(frame, text=f"{text}:")
        label.pack(side="left")
        val_label = ttk.Label(frame, text=str(current_value).ljust(5), width=5)
        val_label.pack(side="right", padx=5)
        slider = ttk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, command=lambda val: self._update_config_slider(param_key, val, val_label))
        slider.set(current_value)
        slider.pack(fill='x', expand=True, padx=5)

    def _add_input(self, parent, text, param_key, decimals):
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=2)
        label = ttk.Label(frame, text=f"{text}:")
        label.pack(side="left")
        current_value = getattr(self.config, param_key)
        entry = ttk.Entry(frame, width=10)
        entry.insert(0, f"{current_value:.{decimals}f}")
        entry.bind("<Return>", lambda event: self._update_config_input(param_key, entry, decimals))
        entry.pack(side="right")

    def _update_config_slider(self, param_key, value_str, val_label):
        value = int(float(value_str))
        setattr(self.config, param_key, value)
        val_label.config(text=str(value).ljust(5))

        if 'K_UNI_SIZE' in param_key:
            size = max(1, value)
            self.config.K_UNI = np.ones((size, size), np.uint8)
        elif 'K_LIMP_SIZE' in param_key:
            size = max(1, value)
            self.config.K_LIMP = np.ones((size, size), np.uint8)
        elif 'PORC_MOS_INT' in param_key:
            self.config.PORC_MOS = value / 100.0
        elif 'SKIP_RATE' in param_key:
            self.config.SKIP_RATE = max(1, value)
        elif 'N_FRAMES_HISTORIAL' in param_key:
            self.config.N_FRAMES_HISTORIAL = max(1, value)
        elif 'MAP_ZOOM_FACTOR' in param_key:
            self.config.MAP_ZOOM_FACTOR = max(1, value)


    def _update_config_input(self, param_key, entry, decimals):
        try:
            value = float(entry.get())
            setattr(self.config, param_key, value)
            entry.delete(0, tk.END)
            entry.insert(0, f"{value:.{decimals}f}")
        except ValueError:
            entry.delete(0, tk.END)
            entry.insert(0, f"{getattr(self.config, param_key):.{decimals}f}")


    def start_processing_thread(self):
        self.thread = ProcesadorEstereoThread(self.config, self, self.cuda_processor)
        self.thread.daemon = True  # Daemon para cerrar inmediatamente
        self.thread.start()

    def pause_thread(self):
        if self.thread and self.thread.is_alive():
            self.thread.pause()
            self.play_button.config(state=tk.NORMAL)

    def resume_thread(self):
        if self.thread and self.thread.is_alive():
            self.thread.resume()
            self.play_button.config(state=tk.DISABLED)
    
    def show_3d_map(self):
        """Abre el visualizador 3D de trayectorias."""
        import threading
        
        def visualize():
            try:
                import open3d as o3d
                import json
                import os
                
                # PRIMERO: Guardar los datos actuales a los archivos JSON
                if hasattr(self, 'thread') and self.thread and self.thread.is_alive():
                    print(f"DEBUG: Guardando datos actuales antes de visualizar...")
                    print(f"  - YOLO en memoria: {len(self.thread.matrices_yolo)} frames")
                    print(f"  - Supervivencia en memoria: {len(self.thread.matrices_supervivencia)} frames")
                    
                    # Guardar datos actuales a disco
                    with open(self.config.OUTPUT_JSON_YOLO, 'w') as f:
                        json.dump(self.thread.matrices_yolo, f)
                    with open(self.config.OUTPUT_JSON_SUPERVIVENCIA, 'w') as f:
                        json.dump(self.thread.matrices_supervivencia, f)
                    print(f"✓ Datos actualizados en archivos JSON")
                else:
                    print(f" No hay thread activo, leyendo archivos JSON existentes...")
                radius_sphere = 0.1
                radius_line = 0.05
                geometries = []
                num_yolo = 0
                num_superv = 0
                num_yolo_valid = 0
                num_superv_valid = 0
                
                # Cargar trayectoria YOLO (verde - para que coincida con el gráfico 2D)
                if os.path.exists(self.config.OUTPUT_JSON_YOLO):
                    with open(self.config.OUTPUT_JSON_YOLO, 'r') as f:
                        poses_yolo = json.load(f)
                    
                    num_yolo = len(poses_yolo)
                    if len(poses_yolo) > 1:
                        points_yolo = []
                        for matrix in poses_yolo:
                            np_matrix = np.array(matrix)
                            position = np_matrix[:3, 3]
                            # Filtrar puntos en el origen (0,0,0) - sin movimiento
                            if not (abs(position[0]) < 0.001 and abs(position[1]) < 0.001 and abs(position[2]) < 0.001):
                                # Amplificar x10 para mejor visualización
                                points_yolo.append(position * 10.0)
                        
                        num_yolo_valid = len(points_yolo)
                        print(f"DEBUG YOLO: {num_yolo} frames totales, {num_yolo_valid} con movimiento (amplificado x10)")
                        
                        if len(points_yolo) > 1:
                            # Crear línea más gruesa usando cilindros entre puntos
                            for i in range(len(points_yolo) - 1):
                                p1 = points_yolo[i]
                                p2 = points_yolo[i + 1]
                                
                                # Crear cilindro entre p1 y p2
                                direction = np.array(p2) - np.array(p1)
                                length = np.linalg.norm(direction)
                                
                                if length > 0.001:  # Solo si hay distancia
                                    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius_line, height=length)
                                    cylinder.paint_uniform_color([0, 1, 0])  # Verde
                                    
                                    # Orientar cilindro
                                    direction_normalized = direction / length
                                    z_axis = np.array([0, 0, 1])
                                    rotation_axis = np.cross(z_axis, direction_normalized)
                                    rotation_angle = np.arccos(np.dot(z_axis, direction_normalized))
                                    
                                    if np.linalg.norm(rotation_axis) > 0.001:
                                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                                        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
                                        cylinder.rotate(R, center=[0, 0, 0])
                                    
                                    cylinder.translate((np.array(p1) + np.array(p2)) / 2)
                                    geometries.append(cylinder)
                            
                            # Punto final YOLO (esfera verde más grande)
                            sphere_yolo = o3d.geometry.TriangleMesh.create_sphere(radius=radius_sphere)
                            sphere_yolo.translate(points_yolo[-1])
                            sphere_yolo.paint_uniform_color([0, 1, 0])
                            geometries.append(sphere_yolo)
                
                # Cargar trayectoria Supervivencia (azul)
                if os.path.exists(self.config.OUTPUT_JSON_SUPERVIVENCIA):
                    with open(self.config.OUTPUT_JSON_SUPERVIVENCIA, 'r') as f:
                        poses_superv = json.load(f)
                    
                    num_superv = len(poses_superv)
                    if len(poses_superv) > 1:
                        points_superv = []
                        for matrix in poses_superv:
                            np_matrix = np.array(matrix)
                            position = np_matrix[:3, 3]
                            # Filtrar puntos en el origen (0,0,0) - sin movimiento
                            if not (abs(position[0]) < 0.001 and abs(position[1]) < 0.001 and abs(position[2]) < 0.001):
                                # Amplificar x10 para mejor visualización
                                points_superv.append(position * 10.0)
                        
                        num_superv_valid = len(points_superv)
                        print(f"DEBUG Supervivencia: {num_superv} frames totales, {num_superv_valid} con movimiento (amplificado x10)")
                        
                        if len(points_superv) > 1:
                            # Crear línea más gruesa usando cilindros entre puntos
                            for i in range(len(points_superv) - 1):
                                p1 = points_superv[i]
                                p2 = points_superv[i + 1]
                                
                                # Crear cilindro entre p1 y p2
                                direction = np.array(p2) - np.array(p1)
                                length = np.linalg.norm(direction)
                                
                                if length > 0.001:  # Solo si hay distancia
                                    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius_line, height=length)
                                    cylinder.paint_uniform_color([0, 0, 1])  # Azul
                                    
                                    # Orientar cilindro
                                    direction_normalized = direction / length
                                    z_axis = np.array([0, 0, 1])
                                    rotation_axis = np.cross(z_axis, direction_normalized)
                                    rotation_angle = np.arccos(np.dot(z_axis, direction_normalized))
                                    
                                    if np.linalg.norm(rotation_axis) > 0.001:
                                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                                        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
                                        cylinder.rotate(R, center=[0, 0, 0])
                                    
                                    cylinder.translate((np.array(p1) + np.array(p2)) / 2)
                                    geometries.append(cylinder)
                            
                            # Punto final Supervivencia (esfera azul más grande)
                            sphere_superv = o3d.geometry.TriangleMesh.create_sphere(radius=radius_sphere)
                            sphere_superv.translate(points_superv[-1])
                            sphere_superv.paint_uniform_color([0, 0, 1])
                            geometries.append(sphere_superv)
                
                # Ejes de coordenadas en el origen (más grandes para visibilidad)
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
                geometries.append(axis)
                
                # Agregar marcadores de detección YOLO (bordes/nudos)
                num_markers = 0
                if hasattr(self, 'thread') and self.thread and hasattr(self.thread, 'yolo_markers'):
                    markers = self.thread.yolo_markers
                    num_markers = len(markers)
                    print(f"  Marcadores YOLO: {num_markers}")
                    
                    # Obtener las matrices YOLO guardadas
                    if os.path.exists(self.config.OUTPUT_JSON_YOLO):
                        with open(self.config.OUTPUT_JSON_YOLO, 'r') as f:
                            matrices = json.load(f)
                        
                        for marker in markers:
                            frame_idx = marker.get('frame_index', -1)
                            
                            # Verificar que el índice sea válido
                            if 0 <= frame_idx < len(matrices):
                                # Obtener posición exacta de la matriz guardada
                                matrix = np.array(matrices[frame_idx])
                                mx = matrix[0, 3] * 10.0  # Ya está en metros, amplificar x10
                                my = matrix[1, 3] * 10.0  # Ya tiene -Y invertido
                                
                                # Color según clase: 0=Borde (rojo), 1=Nudo (magenta)
                                if marker['class'] == 0:
                                    marker_color = [1, 0, 0]  # Rojo (borde)
                                else:
                                    marker_color = [1, 0, 1]  # Magenta (nudo)
                                
                                # Crear esfera para el marcador
                                marker_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.6)
                                marker_sphere.translate([mx, my, 0])
                                marker_sphere.paint_uniform_color(marker_color)
                                geometries.append(marker_sphere)
                                
                                # Crear cilindro vertical como "palo" para mejor visibilidad
                                pole_height = 3.0
                                pole = o3d.geometry.TriangleMesh.create_cylinder(radius=0.1, height=pole_height)
                                pole.translate([mx, my, -pole_height/2])
                                pole.paint_uniform_color(marker_color)
                                geometries.append(pole)
                
                # NO agregar plano - puede confundir la visualización
                
                if geometries:
                    print(f"\n{'='*60}")
                    print(f"Abriendo visor 3D...")
                    print(f"  Verde (YOLO): {num_yolo_valid}/{num_yolo} puntos con movimiento")
                    print(f"  Azul (Supervivencia): {num_superv_valid}/{num_superv} puntos con movimiento")
                    if num_markers > 0:
                        print(f"  🎯 {num_markers} marcadores (Rojo=Borde, Magenta=Nudo)")
                    print(f"{'='*60}\n")
                    o3d.visualization.draw_geometries(
                        geometries, 
                        window_name="Trayectorias 3D - Verde: YOLO | Azul: Supervivencia",
                        width=1280, 
                        height=720,
                        left=100,
                        top=100
                    )
                else:
                    messagebox.showwarning("Sin datos", "No hay datos de tracking para visualizar.\nProcesa primero algunos frames.")
            
            except ImportError:
                messagebox.showerror("Error", "Open3D no está instalado.\nEjecuta: pip install open3d")
            except Exception as e:
                import traceback
                error_msg = f"Error al visualizar mapa 3D:\n{str(e)}\n\n{traceback.format_exc()}"
                print(error_msg)
                messagebox.showerror("Error", error_msg)
        
        # Ejecutar en thread separado para no bloquear GUI
        thread = threading.Thread(target=visualize, daemon=True)
        thread.start()

    def actualizar_gui(self, frame_top: np.ndarray, cns_filt_left_eye: np.ndarray, canv_m: np.ndarray, map_radar: np.ndarray, odometry_graph: np.ndarray, depth_cm: float, pos_m_x: float, pos_m_y: float, global_angle: float, current_frame: int = 0, total_frames: int = 0):

        # Procesar frame principal (vista mono o stereo)
        if self.config.VISTA_MONO:
            # En modo mono, solo mostrar vista izquierda
            h, w = frame_top.shape[:2]
            frame_display = frame_top[:, :w//2].copy()
            # Redimensionar para llenar el espacio completo
            frame_display = cv2.resize(frame_display, (self.VIDEO_WIDTH_FIXED, self.VIDEO_HEIGHT_FIXED))
        else:
            # Modo stereo normal
            frame_display = frame_top
        
        rgb_image = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_image)

        w_l = self.VIDEO_WIDTH_FIXED
        h_l = self.VIDEO_HEIGHT_FIXED

        if w_l > 1 and h_l > 1 and not self.config.VISTA_MONO:
            img_pil = img_pil.resize((w_l, h_l), Image.Resampling.LANCZOS)

        self.tk_image_video = ImageTk.PhotoImage(image=img_pil)
        self.video_label.config(image=self.tk_image_video, text="")

        cns_filt_bgr = cv2.cvtColor(cns_filt_left_eye, cv2.COLOR_GRAY2BGR)
        rgb_mask = cv2.cvtColor(cns_filt_bgr, cv2.COLOR_BGR2RGB)
        img_pil_mask = Image.fromarray(rgb_mask)
        w_m = self.mask_label.winfo_width()
        h_m = self.mask_label.winfo_height()
        if w_m > 1 and h_m > 1:
            img_pil_mask = img_pil_mask.resize((w_m, h_m), Image.Resampling.LANCZOS)

        self.tk_image_mask = ImageTk.PhotoImage(image=img_pil_mask)
        self.mask_label.config(image=self.tk_image_mask, text="")

        rgb_map = cv2.cvtColor(canv_m, cv2.COLOR_BGR2RGB)
        img_pil_map = Image.fromarray(rgb_map)
        w_map = self.mapa_posicion_label.winfo_width()
        h_map = self.mapa_posicion_label.winfo_height()
        if w_map > 1 and h_map > 1:
            img_pil_map = img_pil_map.resize((w_map, h_map), Image.Resampling.LANCZOS)

        self.tk_image_map = ImageTk.PhotoImage(image=img_pil_map)
        self.mapa_posicion_label.config(image=self.tk_image_map, text="")

        rgb_radar = cv2.cvtColor(map_radar, cv2.COLOR_BGR2RGB)
        img_pil_radar = Image.fromarray(rgb_radar)
        w_radar = self.mapa_radar_label.winfo_width()
        h_radar = self.mapa_radar_label.winfo_height()
        if w_radar > 1 and h_radar > 1:
             img_pil_radar = img_pil_radar.resize((w_radar, h_radar), Image.Resampling.LANCZOS)

        self.tk_image_radar = ImageTk.PhotoImage(image=img_pil_radar)
        self.mapa_radar_label.config(image=self.tk_image_radar, text="")
        
        # Mostrar gráfico de odometría visual
        rgb_odometry = cv2.cvtColor(odometry_graph, cv2.COLOR_BGR2RGB)
        img_pil_odometry = Image.fromarray(rgb_odometry)
        w_odometry = self.odometry_label.winfo_width()
        h_odometry = self.odometry_label.winfo_height()
        if w_odometry > 1 and h_odometry > 1:
            img_pil_odometry = img_pil_odometry.resize((w_odometry, h_odometry), Image.Resampling.LANCZOS)

        self.tk_image_odometry = ImageTk.PhotoImage(image=img_pil_odometry)
        self.odometry_label.config(image=self.tk_image_odometry, text="")


        self.depth_label.config(text=f"Profundidad: {depth_cm:.2f} cm" if depth_cm > 0 else "Profundidad: N/A")
        self.pos_label.config(text=f"Posición Global: X: {pos_m_x:.2f} cm, Y: {pos_m_y:.2f} cm")
        self.angle_label.config(text=f"Ángulo: {math.degrees(global_angle):.1f}°")
        
        # Actualizar línea de tiempo
        if total_frames > 0:
            self.timeline_label.config(text=f"Frame: {current_frame} / {total_frames}")
            progress = (current_frame / total_frames) * 100
            self.progress_bar['value'] = progress
        else:
            self.timeline_label.config(text=f"Frame: {current_frame}")
            self.progress_bar['value'] = 0

    def terminar_ejecucion(self):
        """Detiene el thread y guarda todos los datos (reportes, mediciones, imágenes)"""
        print("\n" + "="*60)
        print("FINALIZANDO EJECUCIÓN Y GUARDANDO DATOS...")
        print("="*60)
        
        if self.thread and self.thread.is_alive():
            self.thread.stop()
            self.thread.join(timeout=5.0)
        
        print("✅ Ejecución finalizada y datos guardados correctamente")
        print("="*60 + "\n")
        
        messagebox.showinfo("Finalizado", 
                          "Ejecución terminada.\n\n"
                          "✅ Matrices de odometría guardadas\n"
                          "✅ Mediciones de marcadores guardadas (CSV)\n"
                          "✅ Imágenes capturadas guardadas\n\n"
                          "Revisa la consola para más detalles.")

    def on_closing(self):
        """Cierra la aplicación después de guardar todos los datos"""
        if self.thread and self.thread.is_alive():
            self.thread.stop()
            self.thread.join()
        self.root.destroy()

    def change_video(self):
        # Pausa y detiene el hilo actual de procesamiento
        try:
            if self.thread and self.thread.is_alive():
                self.thread.stop()
                self.thread.join(timeout=2.0)
        except Exception:
            pass

        # Selector de nuevo archivo
        new_file = filedialog.askopenfilename(
            title="Seleccionar nuevo video (MP4 o SVO)",
            filetypes=[("Archivos de Video", "*.mp4 *.avi *.svo"), ("Todos los archivos", "*.*")]
        )
        if not new_file:
            return


        # Actualiza configuración y UI
        self.config.NOM_VID = new_file
        self.config.START_FRAME = 0  # Comenzar desde el inicio
        self.video_label.config(text=f"Cargando: {os.path.basename(self.config.NOM_VID)}", image="")
        self.depth_label.config(text="Profundidad: N/A")
        self.pos_label.config(text="Posición Global: X: 0.0 cm, Y: 0.0 cm")
        self.angle_label.config(text="Ángulo: 0.0°")

        # Reinicia hilo de procesamiento con el nuevo video
        self.start_processing_thread()

    def guardar_reporte(self):
        if not self.thread:
            messagebox.showwarning("Aviso", "No hay procesamiento activo.")
            return

        # Usar la carpeta de sesión del thread
        session_folder = self.thread.session_folder
        
        nombre_archivo = os.path.basename(self.config.NOM_VID)  
        nombre_limpio = os.path.splitext(nombre_archivo)[0]     
        
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        nombre_base = f"Reporte_{nombre_limpio}_{timestamp}"

        # Guardar mapa en la carpeta de sesión
        if self.thread.last_map_radar is not None:
            nombre_img = os.path.join(session_folder, f"{nombre_base}_MAPA.png")
            cv2.imwrite(nombre_img, self.thread.last_map_radar)
        
        # Guardar CSV de daños en la carpeta de sesión
        nombre_csv = os.path.join(session_folder, f"{nombre_base}_DAÑOS.csv")
        try:
            with open(nombre_csv, "w", encoding="utf-8") as f:
                f.write("ID_Daño;Frame;X_Global_cm;Y_Global_cm;Area_px\n")
                
                unique_damages = {}
                
                for dmg in self.thread.damage_log:
                    id_d = dmg['id']
                    fr = dmg['frame']
                    glob_x = dmg.get('global_x', 0.0)
                    glob_y = dmg.get('global_y', 0.0) 
                    area = dmg.get('area', 0) # Si tu detector retorna área
                    gx_str = f"{glob_x:.2f}".replace('.', ',')
                    gy_str = f"{glob_y:.2f}".replace('.', ',')
                    area_str = f"{area:.2f}".replace('.', ',')
                    f.write(f"{id_d};{fr};{gx_str};{gy_str};{area_str}\n")
            
            messagebox.showinfo("Éxito", 
                              f"Reporte guardado en:\n{session_folder}\n\n"
                              f"Archivos:\n- {os.path.basename(nombre_img)}\n- {os.path.basename(nombre_csv)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar el reporte: {str(e)}")
