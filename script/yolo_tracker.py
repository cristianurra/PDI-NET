"""
Módulo de Tracking YOLO para el sistema de procesamiento estéreo.
Basado en odometria_yolov11.py, optimizado para integración con GUI tkinter.
"""

import cv2
import numpy as np
import math
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional

from config import ConfiguracionGlobal


class YOLOTracker:
    """
    Clase para realizar tracking de objetos usando YOLO y calcular vectores de movimiento.
    """
    
    def __init__(self, config: ConfiguracionGlobal):
        """
        Inicializa el tracker YOLO.
        
        Args:
            config: Objeto de configuración global
        """
        self.config = config
        self.model = None
        self.prev_objects: Dict[int, Tuple[float, float]] = {}
        self.prev_x_positions: Dict[int, float] = {}  # Para detectar cruces del centro
        self.is_initialized = False
        
        # Cargar modelo si está habilitado
        if self.config.YOLO_TRACKING_ENABLED:
            self._load_model()
    
    def _load_model(self):
        """Carga el modelo YOLO desde la ruta configurada."""
        try:
            self.model = YOLO(self.config.YOLO_MODEL_PATH)
            
            # Intentar usar CUDA si está disponible
            try:
                import torch
                if torch.cuda.is_available():
                    self.model.to('cuda')
                    print(f"Modelo YOLO cargado desde: {self.config.YOLO_MODEL_PATH} (CUDA)")
                else:
                    print(f"Modelo YOLO cargado desde: {self.config.YOLO_MODEL_PATH} (CPU)")
            except ImportError:
                print(f"Modelo YOLO cargado desde: {self.config.YOLO_MODEL_PATH} (CPU)")
            
            self.is_initialized = True
        except Exception as e:
            print(f"Error al cargar modelo YOLO: {e}")
            self.is_initialized = False
    
    def track_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[float], List[float], List[Dict]]:
        """
        Realiza tracking en un frame y calcula vectores de movimiento.
        
        Args:
            frame: Frame BGR en el que realizar tracking (solo vista izquierda)
            
        Returns:
            - frame_anotado: Frame con tracking dibujado
            - vectors_x: Lista de componentes X de vectores de movimiento
            - vectors_y: Lista de componentes Y de vectores de movimiento
            - detections: Lista de diccionarios con info de detecciones
                         {'class': int, 'name': str, 'cx': float, 'cy': float, 
                          'crossed_center': bool, 'id': int}
        """
        if not self.is_initialized or self.model is None:
            return frame, [], [], []
        
        h, w, _ = frame.shape
        center_img_x, center_img_y = w // 2, h // 2
        
        # Definir tercio central horizontal (33% en cada lado del centro)
        tercio_izq = w * (1/3)
        tercio_der = w * (2/3)
        
        # Realizar tracking con YOLO (solo clases 0 y 1)
        results = self.model.track(
            frame, 
            persist=True, 
            tracker="botsort.yaml", 
            verbose=False, 
            classes=[0, 1]
        )
        
        # Anotar frame
        annotated_frame = results[0].plot()
        
        current_objects = {}
        vectors_x = []
        vectors_y = []
        detections = []
        
        # Nombres de clases (asumiendo: 0=borde, 1=nudo)
        class_names = {0: 'Borde', 1: 'Nudo'}
        
        # Calcular vectores de movimiento
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            detected_list = []
            for box, track_id, cls in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                dist_to_center = math.sqrt((cx - center_img_x)**2 + (cy - center_img_y)**2)
                
                # Detectar si está en el tercio central Y acabó de entrar
                in_center_third = tercio_izq <= cx <= tercio_der
                crossed_center = False
                
                if track_id in self.prev_x_positions:
                    prev_cx = self.prev_x_positions[track_id]
                    # Marcador se activa cuando ENTRA al tercio central (no estaba antes y ahora sí)
                    was_in_center = tercio_izq <= prev_cx <= tercio_der
                    if in_center_third and not was_in_center:
                        crossed_center = True
                elif in_center_third:
                    # Primera vez detectado Y está en el tercio central
                    crossed_center = True
                
                detected_list.append({
                    'id': track_id, 
                    'cx': cx, 
                    'cy': cy, 
                    'dist': dist_to_center,
                    'class': cls,
                    'crossed_center': crossed_center
                })
                
                # Actualizar posición anterior
                self.prev_x_positions[track_id] = cx
            
            # Top 10 objetos más centrales
            detected_list.sort(key=lambda k: k['dist'])
            top_objects = detected_list[:10]
            
            for obj in top_objects:
                tid = obj['id']
                cx, cy = obj['cx'], obj['cy']
                cls = obj['class']
                crossed = obj['crossed_center']
                
                current_objects[tid] = (cx, cy)
                
                # Agregar a lista de detecciones
                detections.append({
                    'class': cls,
                    'name': class_names.get(cls, f'Clase_{cls}'),
                    'cx': cx,
                    'cy': cy,
                    'crossed_center': crossed,
                    'id': tid
                })
                
                if tid in self.prev_objects:
                    prev_cx, prev_cy = self.prev_objects[tid]
                    dx = cx - prev_cx
                    dy = cy - prev_cy
                    vectors_x.append(dx)
                    vectors_y.append(dy)
        
        self.prev_objects = current_objects
        
        return annotated_frame, vectors_x, vectors_y, detections
    
    def reset(self):
        """Reinicia el estado del tracker."""
        self.prev_objects = {}
        self.prev_x_positions = {}


class YOLOOverlayDrawer:
    """
    Clase para dibujar información de tracking YOLO sobre frames existentes.
    """
    
    @staticmethod
    def draw_tracking_boxes(frame: np.ndarray, frame_tracked: np.ndarray, 
                          offset_x: int = 0, offset_y: int = 0,
                          alpha: float = 0.7) -> np.ndarray:
        """
        Dibuja las cajas de tracking de YOLO sobre un frame existente.
        
        Args:
            frame: Frame base sobre el que dibujar
            frame_tracked: Frame con anotaciones de YOLO
            offset_x: Desplazamiento horizontal
            offset_y: Desplazamiento vertical
            alpha: Transparencia (0-1)
            
        Returns:
            Frame con tracking superpuesto
        """
        # Redimensionar frame_tracked si es necesario
        h, w = frame.shape[:2]
        h_t, w_t = frame_tracked.shape[:2]
        
        if (h, w) != (h_t, w_t):
            frame_tracked = cv2.resize(frame_tracked, (w, h))
        
        # Crear máscara de diferencia
        diff = cv2.absdiff(frame[:h_t, :w_t], frame_tracked)
        mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        
        # Aplicar blend solo donde hay anotaciones
        result = frame.copy()
        mask_inv = cv2.bitwise_not(mask)
        
        # Extraer regiones con anotaciones
        fg = cv2.bitwise_and(frame_tracked, frame_tracked, mask=mask)
        bg = cv2.bitwise_and(result[:h_t, :w_t], result[:h_t, :w_t], mask=mask_inv)
        
        # Combinar con transparencia
        combined = cv2.addWeighted(bg, 1.0, fg, alpha, 0)
        result[:h_t, :w_t] = cv2.add(combined, bg)
        
        return result
    
    @staticmethod
    def draw_tracking_info(frame: np.ndarray, vectors_x: List[float], 
                          vectors_y: List[float], pos: Tuple[int, int] = (10, 30),
                          color: Tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
        """
        Dibuja información de tracking en el frame.
        
        Args:
            frame: Frame sobre el que dibujar
            vectors_x: Componentes X de vectores
            vectors_y: Componentes Y de vectores
            pos: Posición del texto
            color: Color del texto
            
        Returns:
            Frame con información dibujada
        """
        result = frame.copy()
        
        n_objects = len(vectors_x)
        avg_vx = np.median(vectors_x) if vectors_x else 0
        avg_vy = np.median(vectors_y) if vectors_y else 0
        
        text = f"YOLO: {n_objects} objs | Vel: ({avg_vx:.1f}, {avg_vy:.1f})"
        cv2.putText(result, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2, cv2.LINE_AA)
        
        return result
