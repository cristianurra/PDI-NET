"""
Módulo de Odometría Visual para el sistema de procesamiento estéreo.
Basado en odometria_yolov11.py, adaptado para mostrar gráfico adaptativo del camino.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional

from config import ConfiguracionGlobal


class VisualOdometry:
    """
    Clase para calcular y visualizar odometría visual basada en tracking de objetos.
    """
    
    def __init__(self, config: ConfiguracionGlobal):
        """
        Inicializa el sistema de odometría visual.
        
        Args:
            config: Objeto de configuración global
        """
        self.config = config
        
        # Estado de la cámara (posición)
        self.cam_pos_x = 0.0
        self.cam_pos_y = 0.0
        
        # Velocidad e inercia
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        
        # Historial de trayectoria (x, y)
        self.trajectory: List[Tuple[float, float]] = []
        
        # Estado del tracking
        self.status = "INICIALIZANDO"
        self.status_color = (200, 200, 200)
    
    def update(self, vectors_x: List[float], vectors_y: List[float]):
        """
        Actualiza la posición de la cámara basándose en vectores de movimiento.
        
        Args:
            vectors_x: Componentes X de vectores de movimiento de objetos (en píxeles)
            vectors_y: Componentes Y de vectores de movimiento de objetos (en píxeles)
        """
        if len(vectors_x) > 0:
            # Calcular movimiento promedio de objetos
            avg_obj_dx = np.median(vectors_x)
            avg_obj_dy = np.median(vectors_y)
            
            # Objeto se mueve derecha -> Cámara se mueve izquierda
            # Convertir píxeles a centímetros usando la misma escala que el tracker
            target_vel_x = -avg_obj_dx * self.config.CM_POR_PX
            target_vel_y = -avg_obj_dy * self.config.CM_POR_PX
            
            # Aplicar aceleración suave
            self.velocity_x = (self.velocity_x * (1 - self.config.YOLO_ACCELERATION) + 
                             target_vel_x * self.config.YOLO_ACCELERATION)
            self.velocity_y = (self.velocity_y * (1 - self.config.YOLO_ACCELERATION) + 
                             target_vel_y * self.config.YOLO_ACCELERATION)
            
            self.status = "TRACKING ACTIVO"
            self.status_color = (0, 255, 0)
        else:
            # Decaimiento de velocidad (fricción)
            self.velocity_x *= self.config.YOLO_FRICTION
            self.velocity_y *= self.config.YOLO_FRICTION
            
            # Detener si velocidad muy baja
            if abs(self.velocity_x) < 0.05:
                self.velocity_x = 0
            if abs(self.velocity_y) < 0.05:
                self.velocity_y = 0
            
            self.status = "INERCIA"
            self.status_color = (0, 255, 255)
        
        # Actualizar posición
        self.cam_pos_x += self.velocity_x
        self.cam_pos_y += self.velocity_y
        
        # Guardar en historial
        self.trajectory.append((self.cam_pos_x, self.cam_pos_y))
    
    def reset(self):
        """Reinicia el estado de la odometría."""
        self.cam_pos_x = 0.0
        self.cam_pos_y = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.trajectory = []
        self.status = "REINICIADO"
        self.status_color = (200, 200, 200)
    
    def get_position(self) -> Tuple[float, float]:
        """Retorna la posición actual de la cámara."""
        return (self.cam_pos_x, self.cam_pos_y)
    
    def get_trajectory(self) -> List[Tuple[float, float]]:
        """Retorna el historial completo de la trayectoria."""
        return self.trajectory


class AdaptiveTrajectoryDrawer:
    """
    Clase para dibujar el gráfico de trayectoria de forma adaptativa.
    El gráfico se ajusta automáticamente para mostrar todo el camino.
    """
    
    def __init__(self, canvas_width: int = 300, canvas_height: int = 300):
        """
        Inicializa el dibujante de trayectoria adaptativa.
        
        Args:
            canvas_width: Ancho del canvas en píxeles
            canvas_height: Alto del canvas en píxeles
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.margin = 30  # Margen alrededor del gráfico
    
    def draw(self, trajectory: List[Tuple[float, float]], 
            current_pos: Tuple[float, float],
            velocity: Tuple[float, float] = (0, 0),
            status: str = "",
            status_color: Tuple[int, int, int] = (200, 200, 200),
            trajectory2: Optional[List[Tuple[float, float]]] = None,
            current_pos2: Optional[Tuple[float, float]] = None,
            markers: Optional[List[dict]] = None) -> np.ndarray:
        """
        Dibuja el gráfico de trayectoria adaptativo.
        
        Args:
            trajectory: Lista de puntos (x, y) de la trayectoria principal (YOLO)
            current_pos: Posición actual de la cámara (YOLO)
            velocity: Velocidad actual (vx, vy)
            status: Texto de estado
            status_color: Color del texto de estado
            trajectory2: Segunda trayectoria (Supervivencia)
            current_pos2: Posición actual segunda trayectoria
            markers: Lista de marcadores de detección {'pos_x', 'pos_y', 'class', 'name', 'marker_id'}
            
        Returns:
            Imagen del gráfico como array numpy
        """
        # Crear canvas negro
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        
        if len(trajectory) == 0:
            # Sin datos, mostrar mensaje
            cv2.putText(canvas, "Sin datos de trayectoria", 
                       (20, self.canvas_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
            return canvas
        
        # Calcular límites de la trayectoria
        traj_x = [p[0] for p in trajectory]
        traj_y = [p[1] for p in trajectory]
        
        # Incluir segunda trayectoria si existe
        if trajectory2:
            traj_x.extend([p[0] for p in trajectory2])
            traj_y.extend([p[1] for p in trajectory2])
        
        min_x, max_x = min(traj_x), max(traj_x)
        min_y, max_y = min(traj_y), max(traj_y)
        
        # Expandir límites ligeramente
        range_x = max_x - min_x
        range_y = max_y - min_y
        
        if range_x < 1.0:
            range_x = 100.0
        if range_y < 1.0:
            range_y = 100.0
        
        # Agregar padding
        padding = 0.1  # 10% de padding
        min_x -= range_x * padding
        max_x += range_x * padding
        min_y -= range_y * padding
        max_y += range_y * padding
        
        range_x = max_x - min_x
        range_y = max_y - min_y
        
        # Calcular escalado
        drawable_width = self.canvas_width - 2 * self.margin
        drawable_height = self.canvas_height - 2 * self.margin
        
        scale = min(drawable_width / range_x, drawable_height / range_y)
        
        def world_to_screen(wx: float, wy: float) -> Tuple[int, int]:
            """Convierte coordenadas del mundo a coordenadas de pantalla."""
            # Normalizar a [0, 1]
            norm_x = (wx - min_x) / range_x
            norm_y = (wy - min_y) / range_y
            
            # Convertir a píxeles (Y positivo hacia arriba en el mundo, hacia abajo en pantalla)
            # Invertimos Y para que arriba en mundo = arriba en pantalla
            sx = int(norm_x * drawable_width + self.margin)
            sy = int(norm_y * drawable_height + self.margin)
            
            return (sx, sy)
        
        # Dibujar ejes de referencia (cruz en el origen si está visible)
        if min_x <= 0 <= max_x and min_y <= 0 <= max_y:
            origin_sx, origin_sy = world_to_screen(0, 0)
            cv2.line(canvas, (origin_sx, 0), (origin_sx, self.canvas_height), 
                    (50, 50, 50), 1)
            cv2.line(canvas, (0, origin_sy), (self.canvas_width, origin_sy), 
                    (50, 50, 50), 1)
        
        # Dibujar trayectoria
        # Optimizar: solo dibujar últimos N puntos si hay muchos
        points_to_draw = trajectory[-1000:] if len(trajectory) > 1000 else trajectory
        
        for i in range(len(points_to_draw) - 1):
            p1 = world_to_screen(points_to_draw[i][0], points_to_draw[i][1])
            p2 = world_to_screen(points_to_draw[i+1][0], points_to_draw[i+1][1])
            
            # Gradiente de color (más reciente = más brillante)
            progress = i / len(points_to_draw)
            color = (0, int(100 + 155 * progress), 0)
            
            cv2.line(canvas, p1, p2, color, 2)
        
        # Dibujar segunda trayectoria (supervivencia) si existe
        if trajectory2 and len(trajectory2) > 1:
            points_to_draw2 = trajectory2[-1000:] if len(trajectory2) > 1000 else trajectory2
            
            for i in range(len(points_to_draw2) - 1):
                p1 = world_to_screen(points_to_draw2[i][0], points_to_draw2[i][1])
                p2 = world_to_screen(points_to_draw2[i+1][0], points_to_draw2[i+1][1])
                
                # Color azul para trayectoria de supervivencia
                progress = i / len(points_to_draw2)
                color = (int(100 + 155 * progress), 0, 0)  # Azul
                
                cv2.line(canvas, p1, p2, color, 2)
            
            # Dibujar posición actual de supervivencia
            if current_pos2:
                curr_sx2, curr_sy2 = world_to_screen(current_pos2[0], current_pos2[1])
                cv2.circle(canvas, (curr_sx2, curr_sy2), 5, (255, 0, 0), -1)  # Azul
                cv2.circle(canvas, (curr_sx2, curr_sy2), 7, (200, 100, 0), 2)
        
        # Dibujar posición actual (punto grande)
        curr_sx, curr_sy = world_to_screen(current_pos[0], current_pos[1])
        cv2.circle(canvas, (curr_sx, curr_sy), 6, (255, 255, 255), -1)
        cv2.circle(canvas, (curr_sx, curr_sy), 8, (0, 255, 255), 2)
        
        # Dibujar flecha de velocidad
        if velocity[0] != 0 or velocity[1] != 0:
            vel_scale = 5
            end_wx = current_pos[0] + velocity[0] * vel_scale
            end_wy = current_pos[1] + velocity[1] * vel_scale
            end_sx, end_sy = world_to_screen(end_wx, end_wy)
            cv2.arrowedLine(canvas, (curr_sx, curr_sy), (end_sx, end_sy), 
                          (0, 0, 255), 2, tipLength=0.3)
        
        # Dibujar marcadores de detección YOLO
        if markers and len(markers) > 0:
            markers_drawn = 0
            markers_skipped = 0
            print(f"🎯 Intentando dibujar {len(markers)} marcadores, trayectoria tiene {len(trajectory)} puntos")
            
            for marker in markers:
                # Obtener posición desde la trayectoria usando el índice del frame
                frame_idx = marker.get('frame_index', -1)
                marker_id = marker.get('marker_id', '?')
                
                # Verificar que el índice sea válido
                if 0 <= frame_idx < len(trajectory):
                    marker_pos = trajectory[frame_idx]
                    mx, my = world_to_screen(marker_pos[0], marker_pos[1])
                    
                    # Verificar que esté dentro del canvas
                    if 0 <= mx < self.canvas_width and 0 <= my < self.canvas_height:
                        # Color según clase: 0=Borde (amarillo), 1=Nudo (magenta)
                        marker_color = (0, 255, 255) if marker['class'] == 0 else (255, 0, 255)
                        
                        # Dibujar círculo con el número del marcador
                        cv2.circle(canvas, (mx, my), 8, marker_color, -1)
                        cv2.circle(canvas, (mx, my), 8, (255, 255, 255), 1)
                        
                        # Dibujar número
                        marker_text = str(marker_id)
                        text_size = cv2.getTextSize(marker_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        text_x = mx - text_size[0] // 2
                        text_y = my + text_size[1] // 2
                        cv2.putText(canvas, marker_text, (text_x, text_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                        cv2.putText(canvas, marker_text, (text_x, text_y),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        markers_drawn += 1
                    else:
                        print(f"⚠️ Marcador {marker_id} fuera de canvas: ({mx}, {my})")
                        markers_skipped += 1
                else:
                    markers_skipped += 1
                    # Debug: imprimir marcadores que no se pueden dibujar
                    print(f"⚠️ Marcador {marker_id} índice inválido: frame_idx={frame_idx}, traj_len={len(trajectory)}")
            
            if markers_drawn > 0:
                print(f"✅ Dibujados {markers_drawn} marcadores ({markers_skipped} saltados)")
        
        # Dibujar información
        cv2.putText(canvas, f"Pos: ({current_pos[0]:.1f}, {current_pos[1]:.1f})", 
                   (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, f"Rango X: {range_x:.1f} | Y: {range_y:.1f}", 
                   (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Leyenda
        legend_y = 60
        cv2.putText(canvas, "Verde: YOLO", (5, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        legend_y += 20
        if trajectory2:
            cv2.putText(canvas, "Azul: Superv.", (5, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            legend_y += 20
        if markers and len(markers) > 0:
            cv2.putText(canvas, "Rojo: Marcador", (5, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            legend_y += 18
            cv2.putText(canvas, "Amarillo: Borde", (5, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            legend_y += 18
            cv2.putText(canvas, "Magenta: Nudo", (5, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
        
        cv2.putText(canvas, status, 
                   (5, self.canvas_height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # Dibujar borde
        cv2.rectangle(canvas, (0, 0), (self.canvas_width-1, self.canvas_height-1), 
                     (80, 80, 80), 1)
        
        return canvas
    
    def resize_canvas(self, width: int, height: int):
        """Cambia el tamaño del canvas."""
        self.canvas_width = width
        self.canvas_height = height
