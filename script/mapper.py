import cv2
import numpy as np
import math

class GlobalMapper2D:
    def __init__(self, center_x=500, center_y=500):
        self.map_size = 1000
        self.global_x = float(center_x)
        self.global_y = float(center_y)
        self.global_angle = 0.0 # Radianes (0 = Mirando arriba)

        self.trayectoria = [] 
        self.scale = 2.0 
        
    def update_position(self, tracked_objects):
        # Necesitamos al menos 3 puntos para calcular una transformación afín confiable
        if len(tracked_objects) < 3: return
        # 1. Obtener pares de puntos (Anterior -> Actual)
        prev_pts = []
        curr_pts = []
        
        for obj in tracked_objects:
            if len(obj['hist_vel']) > 0:
                curr = obj['pos']
                vel = obj['hist_vel'][-1]
                # Reconstruir posición anterior: pos - velocidad
                prev = (curr[0] - vel[0], curr[1] - vel[1])
                
                prev_pts.append(prev)
                curr_pts.append(curr)
        
        if len(prev_pts) < 3: return
        
        # Conversión a numpy para OpenCV
        prev_np = np.array(prev_pts, dtype=np.float32).reshape(-1, 1, 2)
        curr_np = np.array(curr_pts, dtype=np.float32).reshape(-1, 1, 2)
        
        # 2. Calcular Matriz de Transformación (Rotación + Traslación)
        # RANSAC elimina puntos erróneos (ruido que no es malla)
        M, inliers = cv2.estimateAffinePartial2D(curr_np, prev_np)
        
        if M is not None:
            # Extraer rotación y traslación de la matriz M
            # M = [[cos, -sin, tx], [sin, cos, ty]]
            dx_local = M[0, 2]
            dy_local = M[1, 2]
            d_angle = np.arctan2(M[1, 0], M[0, 0])
            
            # --- NUEVO: LIMITADOR DE VELOCIDAD ---
            dist_sq = dx_local**2 + dy_local**2
            
            # Si el movimiento es absurdo (> 30px por frame), lo ignoramos (es ruido)
            if dist_sq > 900: 
                dx_local = 0
                dy_local = 0
                d_angle = 0
            
            # Factor de 'Cámara Lenta' para el mapa (El ROV es lento)
            speed_k = 0.05  # <--- BAJAMOS ESTO MUCHO
            
            self.global_angle += d_angle * 0.5 # También suavizamos el giro
            
            # 4. Rotar el movimiento local al sistema global
            # (Moverse "al frente" depende de hacia dónde mira el ROV)
            c, s = np.cos(self.global_angle), np.sin(self.global_angle)
            
            # Rotación de vector 2D
            dx_global = (dx_local * c - dy_local * s) * speed_k
            dy_global = (dx_local * s + dy_local * c) * speed_k
            
            # --- NUEVO: NO SALIRSE DEL MAPA ---
            new_x = self.global_x + dx_global
            new_y = self.global_y + dy_global
            
            # Solo actualizamos si estamos dentro del margen (con un borde de 50px)
            if 50 < new_x < self.map_size - 50:
                self.global_x = new_x
            if 50 < new_y < self.map_size - 50:
                self.global_y = new_y
            
            self.trayectoria.append((int(self.global_x), int(self.global_y)))

    def draw_map(self, tracked_objects, is_quality_good=True):
        canvas = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        
        # Grid
        for i in range(0, self.map_size, 100):
            cv2.line(canvas, (i, 0), (i, self.map_size), (30, 30, 30), 1)
            cv2.line(canvas, (0, i), (self.map_size, i), (30, 30, 30), 1)
        # Trayectoria
        if len(self.trayectoria) > 1:
            pts = np.array(self.trayectoria, np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts], False, (0, 255, 255), 2)
        # DIBUJAR ROV ROTADO (Triángulo)
        cx, cy = int(self.global_x), int(self.global_y)
        # Puntos base del triángulo mirando hacia "Arriba" (negativo Y)
        tip = np.array([0, -15])
        bl = np.array([-10, 10])
        br = np.array([10, 10])
        
        # Matriz de rotación para el dibujo
        c, s = np.cos(self.global_angle), np.sin(self.global_angle)
        rot_mat = np.array([[c, -s], [s, c]])
        
        # Rotar puntos y trasladar al centro
        p1 = np.dot(rot_mat, tip) + [cx, cy]
        p2 = np.dot(rot_mat, bl) + [cx, cy]
        p3 = np.dot(rot_mat, br) + [cx, cy]
        
        pts_rov = np.array([p1, p2, p3], np.int32)
        cv2.drawContours(canvas, [pts_rov], 0, (0, 0, 255), -1) # Rojo = ROV
        
        # Indicador de frente (Línea azul corta)
        p_nose = np.dot(rot_mat, np.array([0, -25])) + [cx, cy]
        cv2.line(canvas, (cx, cy), (int(p_nose[0]), int(p_nose[1])), (255, 0, 0), 2)
        # Muros (Proyectados según rotación)
        if is_quality_good and tracked_objects:
            for obj in tracked_objects:
                u, v = obj['pos']
                z = obj.get('depth_cm', 0)
                if z > 0:
                    # Posición relativa al ROV (Cámara)
                    # x lateral, y profundidad (z)
                    local_x = (u - 640) * 0.5 
                    local_y = -z * self.scale # Negativo porque "al frente" es arriba en el mapa local
                    
                    # Rotar al mundo global
                    world_x = local_x * c - local_y * s
                    world_y = local_x * s + local_y * c
                    
                    wx = int(cx + world_x)
                    wy = int(cy + world_y)
                    
                    if 0 <= wx < self.map_size and 0 <= wy < self.map_size:
                        color = obj.get('color', (200,200,200))
                        cv2.circle(canvas, (wx, wy), 2, color, -1)
        # Estado
        if is_quality_good:
            cv2.putText(canvas, f"ANG: {math.degrees(self.global_angle):.1f} deg", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(canvas, "ESPERANDO DATOS...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        return canvas
