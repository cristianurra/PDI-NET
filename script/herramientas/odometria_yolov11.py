import sys
import pyzed.sl as sl
import cv2
import numpy as np
import math
from ultralytics import YOLO
from collections import deque
import json

# --- CONFIGURACIÓN ---
PATH_SVO = "D:\\Universidad\\Semestre Actual\\TEL328 - Procesamiento digital de imagenes\\PDI\\RepoC_Urra\\PDI-NET.last\\videos\\2025_09_25_12_47_04.svo"
PATH_MODELO_YOLO = "models\\best.pt"
OUTPUT_JSON = "odometria_yolo.json"

# Radar Visual (Ventana)
RADAR_SIZE = 600
RADAR_CENTER = RADAR_SIZE // 2
SCALE_FACTOR = 1.5 # Ajusta sensibilidad

# Física (Inercia)
FRICTION = 0.95    
ACCELERATION = 0.2 

def main():
    print("Cargando modelo YOLO...")
    model = YOLO(PATH_MODELO_YOLO)

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(PATH_SVO)
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE 
    init_params.svo_real_time_mode = False

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Error al abrir SVO")
        sys.exit(1)

    runtime = sl.RuntimeParameters()
    image_zed = sl.Mat()

    # --- VARIABLES DE ESTADO ---
    cam_pos_x = 0.0
    cam_pos_y = 0.0
    
    velocity_x = 0.0
    velocity_y = 0.0

    prev_objects = {}
    
    # Historial completo para el Radar y el JSON
    # Guardamos tuplas: (frame_idx, x, y)
    full_trajectory = [] 
    
    # Lista para exportar a Open3D (Matrices 4x4)
    matrices_4x4_export = []

    print("Iniciando Odometría Visual Infinita... (Q para salir)")
    
    key = ''
    frame_idx = 0

    while key != 113:
        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            frame_idx += 1
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame_bgra = image_zed.get_data()
            frame_bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
            h, w, _ = frame_bgr.shape
            center_img_x, center_img_y = w // 2, h // 2

            # 1. TRACKING
            # results = model.track(frame_bgr, persist=True, tracker="botsort.yaml", verbose=False)
            results = model.track(
                frame_bgr, 
                persist=True, 
                tracker="botsort.yaml", 
                verbose=False, 
                classes=[0, 1]  # <--- ESTO ES LO NUEVO
            )
            annotated_frame = results[0].plot()

            current_objects = {}
            vectors_x = []
            vectors_y = []

            # 2. CALCULO DE VECTORES
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                detected_list = []
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    dist_to_center = math.sqrt((cx - center_img_x)**2 + (cy - center_img_y)**2)
                    detected_list.append({'id': track_id, 'cx': cx, 'cy': cy, 'dist': dist_to_center})

                # Top 10 objetos más centrales
                detected_list.sort(key=lambda k: k['dist'])
                top_objects = detected_list[:10] 

                for obj in top_objects:
                    tid = obj['id']
                    cx, cy = obj['cx'], obj['cy']
                    current_objects[tid] = (cx, cy)

                    if tid in prev_objects:
                        prev_cx, prev_cy = prev_objects[tid]
                        dx = cx - prev_cx
                        dy = cy - prev_cy
                        vectors_x.append(dx)
                        vectors_y.append(dy)

            # 3. FÍSICA (VELOCIDAD E INERCIA)
            if len(vectors_x) > 0:
                avg_obj_dx = np.median(vectors_x)
                avg_obj_dy = np.median(vectors_y)

                # Objeto se mueve derecha -> Cámara se mueve izquierda
                target_vel_x = -avg_obj_dx * SCALE_FACTOR
                target_vel_y = -avg_obj_dy * SCALE_FACTOR # Y hacia abajo en pantalla es +

                velocity_x = velocity_x * (1 - ACCELERATION) + target_vel_x * ACCELERATION
                velocity_y = velocity_y * (1 - ACCELERATION) + target_vel_y * ACCELERATION
                status_text = "TRACKING ACTIVO"
                color_status = (0, 255, 0)
            else:
                # Decaimiento
                velocity_x *= FRICTION
                velocity_y *= FRICTION
                if abs(velocity_x) < 0.05: velocity_x = 0
                if abs(velocity_y) < 0.05: velocity_y = 0
                status_text = "INERCIA"
                color_status = (0, 255, 255)

            # 4. ACTUALIZAR POSICIÓN
            cam_pos_x += velocity_x
            cam_pos_y += velocity_y 

            prev_objects = current_objects
            
            # Guardar en historial
            full_trajectory.append((cam_pos_x, cam_pos_y))

            # --- 5. GENERAR MATRIZ 4x4 PARA JSON ---
            # Construimos una matriz de identidad
            # [ 1 0 0 X ]
            # [ 0 1 0 Y ]
            # [ 0 0 1 Z ]
            # [ 0 0 0 1 ]
            mat_4x4 = np.eye(4)
            
            # Asignamos traslación. 
            # IMPORTANTE: Escalamos (dividir por 100) para que 100 pixeles sean 1 metro en Open3D aprox.
            # Invertimos Y para que coincida con coordenadas cartesianas estándar (Arriba es positivo)
            mat_4x4[0, 3] = cam_pos_x / 100.0  
            mat_4x4[1, 3] = -cam_pos_y / 100.0 
            mat_4x4[2, 3] = 0.0 # Z quieto como pediste
            
            matrices_4x4_export.append(mat_4x4.tolist())

            # --- 6. DIBUJAR RADAR (MODO INFINITO / CÁMARA CENTRADA) ---
            radar_img = np.zeros((RADAR_SIZE, RADAR_SIZE, 3), dtype=np.uint8)
            
            # Dibujar Ejes Estáticos (Cruz central)
            cv2.line(radar_img, (RADAR_CENTER, 0), (RADAR_CENTER, RADAR_SIZE), (30, 30, 30), 1)
            cv2.line(radar_img, (0, RADAR_CENTER), (RADAR_SIZE, RADAR_CENTER), (30, 30, 30), 1)

            # Dibujar CAMINO HISTÓRICO relativo a la cámara
            # Lógica: Punto_Dibujo = (Punto_Histórico - Posición_Cámara_Actual) + Centro_Pantalla
            # Esto hace que el "mundo" se mueva y la cámara se quede quieta
            
            # Para optimizar, solo dibujamos los últimos 500 puntos si la lista es muy larga
            puntos_a_dibujar = full_trajectory[-500:] if len(full_trajectory) > 500 else full_trajectory

            for hx, hy in puntos_a_dibujar:
                rel_x = int(hx - cam_pos_x) + RADAR_CENTER
                rel_y = int(hy - cam_pos_y) + RADAR_CENTER
                
                # Solo dibujamos si cae dentro de la ventana visible
                if 0 <= rel_x < RADAR_SIZE and 0 <= rel_y < RADAR_SIZE:
                    radar_img[rel_y, rel_x] = (0, 255, 0) # Verde

            # Dibujar CÁMARA (Siempre en el centro)
            cv2.circle(radar_img, (RADAR_CENTER, RADAR_CENTER), 6, (255, 255, 255), -1)
            
            # Flecha de velocidad
            end_x = RADAR_CENTER + int(velocity_x * 5)
            end_y = RADAR_CENTER + int(velocity_y * 5)
            cv2.arrowedLine(radar_img, (RADAR_CENTER, RADAR_CENTER), (end_x, end_y), (0, 0, 255), 2)

            # Textos
            cv2.putText(radar_img, f"Pos: {cam_pos_x:.1f}, {cam_pos_y:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(radar_img, status_text, (10, RADAR_SIZE - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_status, 2)

            cv2.imshow("Tracking YOLO", annotated_frame)
            cv2.imshow("Odometria Visual (Mapa Dinamico)", radar_img)
            
            key = cv2.waitKey(10)

        elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            break

    # --- GUARDAR JSON AL SALIR ---
    print(f"Guardando {len(matrices_4x4_export)} puntos de trayectoria en {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(matrices_4x4_export, f)
        
    print("¡Archivo guardado exitosamente!")
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()