import os
import cv2
import numpy as np
import math
import time
import threading
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


class ProcesadorEstereoThread(threading.Thread):
    def __init__(self, config: ConfiguracionGlobal, gui_ref):
        super().__init__()
        self.config = config
        self.gui_ref = gui_ref
        self._running = True
        self.is_paused = False
        self.mapeo = GlobalMapper2D(config)
        self.hist_celdas_vis: Dict[Tuple[int, int], Tuple[float, np.ndarray]] = {}
        self.tracked_objects_history: List[List[Dict[str, Any]]] = []
        self.damage_detector = DamageDetector(config)

    def stop(self):
        self._running = False

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def run(self):
        ext = os.path.splitext(self.config.NOM_VID)[1].lower()
        is_svo = ext == '.svo'
        frame_generator = None
        cap = None

        if is_svo:
            frame_generator, total_frames, w, h = open_svo_file(self.config.NOM_VID)
            if frame_generator is None or w == 0:
                self.gui_ref.root.after(0, lambda: messagebox.showerror("Error", "Error al abrir."))
                return
        else:
            cap = cv2.VideoCapture(self.config.NOM_VID)
            if not cap.isOpened():
                self.gui_ref.root.after(0, lambda: messagebox.showerror("Error", f"No se pudo abrir el video: {self.config.NOM_VID}"))
                return
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            cap.set(cv2.CAP_PROP_POS_FRAMES, self.config.START_FRAME)

        if w == 0 or h == 0:
            self.gui_ref.root.after(0, lambda: messagebox.showerror("Error", "El video tiene dimensiones inválidas."))
            return

        q_w, q_h = w // self.config.Q_X, h // self.config.Q_Y

        tracker = Tracker(self.config.UMB_DIST, self.config.N_VEL_PR, self.config)
        pos_m_x, pos_m_y = 0.0, 0.0

        ret, frame = (next(frame_generator) if is_svo else cap.read())
        frame_counter = 0

        while self._running and ret:

            while self.is_paused and self._running:
                time.sleep(0.1)

            if not self._running:
                break

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
                frame_with_damages, damages_info = self.damage_detector.detect(frame_left)  
                frame_top[:, :w//2] = frame_with_damages

                
                dib_ayu(frame_top, w, h, q_w, q_h, self.config)
                del_p_x, del_p_y, vista_actual_limpia = dib_mov(frame_top, objs, w, h, depth_cm, self.config)
                dib_escala_profundidad(frame_top, w, h, self.config)

                
                
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

                rect_sz_cm_actual = self.config.RECT_SZ_CM_FALLBACK
                if depth_cm > 0:
                    ancho_total_cm_proyectado_ref = (depth_cm * (w / 2)) / self.config.FOCAL_PIX
                    rect_sz_cm_actual = rect_sz_cm_actual / self.config.Q_X
                    rect_sz_cm_actual = np.clip(rect_sz_cm_actual, 10.0, 100.0)

                    grid_x = round(pos_m_x / self.config.FIXED_GRID_SIZE_CM)
                    grid_y = round(pos_m_y / self.config.FIXED_GRID_SIZE_CM)
                    celda_id = (grid_x, grid_y)
                    normalized_view = normalize_cell_view(vista_actual_limpia.copy(), cell_target_size=(100, 100))

                    if celda_id not in self.hist_celdas_vis or depth_cm < self.hist_celdas_vis[celda_id][0]:
                        existing_image = self.hist_celdas_vis.get(celda_id, (None, None))[1]
                        registered_image = register_image_to_map(normalized_view, existing_image)
                        self.hist_celdas_vis[celda_id] = (depth_cm, registered_image)

                cns_filt_left_eye = cns_filt[:, :w // 2]
                map_display_w = 400
                map_display_h = 400

                map_radar = self.mapeo.draw_map(objs, frames_history=self.tracked_objects_history)

                canv_m = dib_map(
                    self.hist_celdas_vis, pos_m_x, pos_m_y, self.config.FIXED_GRID_SIZE_CM,
                    rect_sz_cm_actual, map_display_w, map_display_h,
                    self.config.FIXED_GRID_SIZE_CM, self.config.FIXED_GRID_SIZE_CM, self.config
                )

                self.gui_ref.root.after(0, self.gui_ref.actualizar_gui, frame_top, cns_filt_left_eye, canv_m, map_radar, depth_cm, pos_m_x, pos_m_y, self.mapeo.global_angle)

            frame_counter += 1

            time.sleep(0.001)

            ret, frame = (next(frame_generator) if is_svo else cap.read())

        if cap: cap.release()
        self.gui_ref.root.after(0, lambda: self.gui_ref.depth_label.config(text="Procesamiento TERMINADO."))


class StereoAppTkinter:
    VIDEO_WIDTH_FIXED = 1440
    VIDEO_HEIGHT_FIXED = 480

    def __init__(self, root: tk.Tk, config: ConfiguracionGlobal):
        self.root = root
        self.config = config
        self.root.title("Sistema de Procesamiento Estéreo (Tkinter)")

        try:
            self.root.state('zoomed')
        except tk.TclError:
            width = self.root.winfo_screenwidth()
            height = self.root.winfo_screenheight()
            self.root.geometry(f"{width}x{height}+0+0")
            self.root.state('normal')

        self.thread: ProcesadorEstereoThread = None
        self.tk_image_video = None
        self.tk_image_mask = None
        self.tk_image_map = None
        self.tk_image_radar = None

        self.style = ttk.Style()
        self.style.theme_use('clam')

        self.var_profundidad_stereo = tk.BooleanVar(value=self.config.PROFUNDIDAD_STEREO_ACTIVA)

        self._select_file_and_start()


    def _select_file_and_start(self):
        self.root.withdraw()

        file_path = filedialog.askopenfilename(
            title="1. Seleccionar Video Estéreo (MP4 o SVO)",
            filetypes=[("Archivos de Video", "*.mp4 *.avi *.svo"), ("Todos los archivos", "*.*")]
        )

        if not file_path:
            self.root.quit()
            return

        while True:
            try:
                start_frame_str = simpledialog.askstring("2. Frame Inicial",
                                                         "Ingrese el frame desde el que desea iniciar el procesamiento (Ej: 0, 500):",
                                                         initialvalue="500", #Comienza por defecto en 500 para eliminar los primeros frame de adaptacion y calibración de la camara.
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

        self.mapa_radar_label = ttk.Label(video_col_frame, text="Mapa 3D (Radar)", background="#000", foreground="#fff")
        self.mapa_radar_label.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

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

        ttk.Button(control_col_frame, text="🔴 Terminar Ejecución", command=self.on_closing, style='Danger.TButton').grid(row=3, column=0, sticky="ew", padx=5, pady=5)

        # Botón para cambiar el video
        ttk.Button(control_col_frame, text="🔄 Cambiar Video", command=self.change_video).grid(row=4, column=0, sticky="ew", padx=5, pady=5)

        self.style.configure('Danger.TButton', foreground='red', font=('Helvetica', 10, 'bold'))

    def _create_control_panel(self, parent):
        group = ttk.LabelFrame(parent, text="Ajuste de Parámetros de Tracking", padding="10")

        switch_frame = ttk.Frame(group)
        switch_frame.pack(fill='x', pady=5)

        ttk.Label(switch_frame, text="Profundidad Estéreo:").pack(side="left", padx=(10, 0))
        ttk.Checkbutton(switch_frame, variable=self.var_profundidad_stereo, command=self._update_switches, style='TCheckbutton').pack(side="left", padx=5)

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
        self.thread = ProcesadorEstereoThread(self.config, self)
        self.thread.daemon = True
        self.thread.start()

    def pause_thread(self):
        if self.thread and self.thread.is_alive():
            self.thread.pause()
            self.play_button.config(state=tk.NORMAL)

    def resume_thread(self):
        if self.thread and self.thread.is_alive():
            self.thread.resume()
            self.play_button.config(state=tk.DISABLED)

    def actualizar_gui(self, frame_top: np.ndarray, cns_filt_left_eye: np.ndarray, canv_m: np.ndarray, map_radar: np.ndarray, depth_cm: float, pos_m_x: float, pos_m_y: float, global_angle: float):

        rgb_image = cv2.cvtColor(frame_top, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(rgb_image)

        w_l = self.VIDEO_WIDTH_FIXED
        h_l = self.VIDEO_HEIGHT_FIXED

        if w_l > 1 and h_l > 1:
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


        self.depth_label.config(text=f"Profundidad: {depth_cm:.2f} cm" if depth_cm > 0 else "Profundidad: N/A")
        self.pos_label.config(text=f"Posición Global: X: {pos_m_x:.2f} cm, Y: {pos_m_y:.2f} cm")
        self.angle_label.config(text=f"Ángulo: {math.degrees(global_angle):.1f}°")


    def on_closing(self):
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

        # Solicita nuevo frame inicial
        start_frame = self.config.START_FRAME
        try:
            start_str = simpledialog.askstring(
                "Frame Inicial",
                "Ingrese el frame inicial (Ej: 0, 500):",
                initialvalue=str(start_frame),
                parent=self.root,
            )
            if start_str is None:
                return
            start_frame = int(start_str)
            if start_frame < 0:
                messagebox.showerror("Error", "El frame inicial debe ser un número positivo.")
                return
        except ValueError:
            messagebox.showerror("Error", "Entrada inválida. Debe ser un número entero.")
            return

        # Actualiza configuración y UI
        self.config.NOM_VID = new_file
        self.config.START_FRAME = start_frame
        self.video_label.config(text=f"Cargando: {os.path.basename(self.config.NOM_VID)}", image="")
        self.depth_label.config(text="Profundidad: N/A")
        self.pos_label.config(text="Posición Global: X: 0.0 cm, Y: 0.0 cm")
        self.angle_label.config(text="Ángulo: 0.0°")

        # Reinicia hilo de procesamiento con el nuevo video
        self.start_processing_thread()
