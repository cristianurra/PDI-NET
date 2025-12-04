"""
Módulo de configuración global del sistema de procesamiento estéreo.
Define todos los parámetros y constantes utilizados en el sistema.
"""

import numpy as np
import cv2
import os

class ConfiguracionGlobal:
    """
    Clase de configuración global que centraliza todos los parámetros del sistema.
    Incluye configuraciones para tracking, procesamiento estéreo, visualización y YOLO.
    """
    
    def __init__(self, nom_vid=""):
        """
        Inicializa la configuración global del sistema.
        
        Args:
            nom_vid: Nombre del archivo de video a procesar (opcional)
        """
        # ===== CONFIGURACIÓN DE VIDEO =====
        self.NOM_VID = nom_vid  # Nombre del archivo de video de entrada
        
        # ===== PARÁMETROS DE TRACKING =====
        self.RAD_PUN = 6  # Radio de los puntos de tracking en píxeles
        self.UMB_DIST = 75  # Umbral de distancia máxima para asociar puntos entre frames (píxeles)
        self.N_VEL_PR = 10  # Número de frames para calcular velocidad promedio
        self.MIN_SUPERVIVENCIA_FR = 4  # Mínimo de frames que debe sobrevivir un punto para ser considerado válido
        self.FRAMES_MAX_ESTATICO = 3  # Máximo de frames sin movimiento antes de considerar punto estático
        
        # ===== PARÁMETROS DE GRILLA Y ESCALA =====
        self.Q_X = 6  # Número de celdas en X para subdivisión de imagen
        self.Q_Y = 5  # Número de celdas en Y para subdivisión de imagen
        self.SEP_CM = 2.5  # Separación real en centímetros entre puntos de referencia
        self.SEP_PX_EST = 20  # Separación estimada en píxeles entre puntos
        self.CM_POR_PX = self.SEP_CM / self.SEP_PX_EST  # Factor de conversión centímetros por píxel

        # ===== PARÁMETROS DE CÁMARA ESTÉREO =====
        self.BASELINE_CM = 12.0  # Distancia entre cámaras en centímetros (baseline estéreo)
        self.FOCAL_PIX = 800.0  # Longitud focal en píxeles
        self.MIN_DEPTH_CM = 20.0  # Profundidad mínima detectable en centímetros
        self.MAX_DEPTH_CM = 300.0  # Profundidad máxima detectable en centímetros
        self.N_DEPTH_PR = 5  # Número de muestras para promedio de profundidad
        self.MIN_DISPARITY = 5  # Disparidad mínima en píxeles
        self.MAX_DISPARITY = 150  # Disparidad máxima en píxeles
        self.Y_TOLERANCE = 6  # Tolerancia en Y para matching estéreo (píxeles)

        # ===== PARÁMETROS DE VISUALIZACIÓN =====
        self.ESC_VEC = 20  # Escala para dibujar vectores de movimiento
        self.C_CAM = (0, 255, 255)  # Color de la cámara (amarillo en BGR)
        self.FIXED_GRID_SIZE_CM = 40.0  # Tamaño fijo de celda de grilla en centímetros
        self.RECT_SZ_CM_FALLBACK = 30.0  # Tamaño de rectángulo de captura por defecto (cm)
        self.RECT_MARGIN_CM = 5.0  # Margen alrededor del rectángulo de captura (cm)
        self.MAP_PAD_PX = 40  # Padding del mapa en píxeles
        self.MAP_ESC_V = 5.0  # Escala de visualización del mapa (píxeles por centímetro)
        self.C_MAP_TXT = (255, 255, 255)  # Color de texto en mapa (blanco)
        self.C_MAP_ACT = (0, 0, 255)  # Color de zona activa en mapa (rojo)
        self.C_NARAN = (0, 165, 255)  # Color naranja para marcadores (BGR)
        self.C_GRIS = (100, 100, 100)  # Color gris para elementos inactivos
        self.C_ACT = (0, 255, 0)  # Color verde para elementos activos
        self.C_DANO = (0, 0, 255)  # Color rojo para indicar daños

        # ===== PARÁMETROS DE PROCESAMIENTO DE MALLA =====
        self.MESH_CONSOLIDATE_K = 7  # Tamaño del kernel para consolidación de malla
        self.Y_MASK_OFFSET = 100  # Offset en Y para máscara de segmentación (píxeles)

        # ===== KERNELS MORFOLÓGICOS =====
        self.K_UNI_SIZE = 5  # Tamaño del kernel de uniformización
        self.K_LIMP_SIZE = 3  # Tamaño del kernel de limpieza
        self.K_VERT_FILL_H = 31  # Altura del kernel de relleno vertical
        self.K_VERT_FILL_W = 3  # Ancho del kernel de relleno vertical

        self.K_UNI = np.ones((self.K_UNI_SIZE, self.K_UNI_SIZE), np.uint8)  # Kernel de uniformización
        self.K_LIMP = np.ones((self.K_LIMP_SIZE, self.K_LIMP_SIZE), np.uint8)  # Kernel de limpieza
        self.K_VERT_FILL = cv2.getStructuringElement(cv2.MORPH_RECT, (self.K_VERT_FILL_W, self.K_VERT_FILL_H))  # Kernel para rellenar verticalmente

        # ===== CONFIGURACIÓN DE CELDAS ACTIVAS =====
        self.Q_ACT_BASE = [(1, 1), (1, 4), (2, 1), (2, 4), (3, 1), (3, 4)]  # Celdas activas base (índices de grilla)
        self.PORC_MOS_INT = 100  # Porcentaje de mosaico interno (no usado actualmente)
        self.PORC_MOS = 1.0  # Factor de porcentaje de mosaico
        self.EDGE_POINT_RADIUS = 2  # Radio de puntos de borde en píxeles

        # ===== DETECCIÓN DE MARCADORES NARANJAS =====
        self.ORANGE_HSV_LOW = (5, 120, 150)  # Rango HSV inferior para detección de naranja
        self.ORANGE_HSV_HIGH = (22, 255, 255)  # Rango HSV superior para detección de naranja
        self.ORANGE_MIN_AREA = 30  # Área mínima de marcador naranja (píxeles²)
        self.ORANGE_MAX_AREA = 5000  # Área máxima de marcador naranja (píxeles²)
        self.ORANGE_CIRCULARITY = 0.4  # Umbral de circularidad para validar marcadores

        # ===== CONFIGURACIÓN DE MAPA Y FRAMES =====
        self.MAP_CAN_SZ = 800  # Tamaño del canvas del mapa en píxeles
        self.START_FRAME = 0  # Frame inicial para comenzar procesamiento

        # ===== CONFIGURACIÓN DE PROCESAMIENTO =====
        self.PROFUNDIDAD_STEREO_ACTIVA = True  # Activar/desactivar cálculo de profundidad estéreo
        self.SKIP_RATE = 1  # Tasa de salto de frames (1 = procesar todos)
        self.N_FRAMES_HISTORIAL = 5  # Número de frames a mantener en historial
        self.MAP_ZOOM_FACTOR = 10  # Factor de zoom del mapa

        # ===== DETECCIÓN DE DAÑOS =====
        self.DMG_ALPHA = 0.1  # Factor alpha para suavizado exponencial en detección de daños
        self.DMG_NUM_NB = 4  # Número de vecinos para comparación de daños
        self.DMG_FRAMES = 6  # Frames necesarios para confirmar daño
        self.DMG_THRESHOLD = 2.5  # Umbral de desviación estándar para detectar anomalías
        self.DMG_DIST_TRACK = 20  # Distancia máxima para tracking de daños (cm)
        
        # ===== CONFIGURACIÓN DE VISTA Y YOLO =====
        self.VISTA_MONO = False  # False = vista estéreo, True = vista monocular
        self.YOLO_MODEL_PATH = "models/best.pt"  # Ruta al modelo YOLO entrenado
        self.YOLO_TRACKING_ENABLED = True  # Activar/desactivar tracking YOLO
        self.YOLO_SCALE_FACTOR = 1.5  # Factor de escala para visualización YOLO
        self.YOLO_FRICTION = 0.95  # Factor de fricción para odometría YOLO (0-1)
        self.YOLO_ACCELERATION = 0.2  # Factor de aceleración para odometría YOLO (0-1)
        
        # ===== CONFIGURACIÓN DE VISUALIZACIÓN DE VECTORES =====
        self.MOSTRAR_VECTOR_SUPERVIVENCIA = True  # Mostrar vectores de método supervivencia
        self.MOSTRAR_VECTOR_YOLO = True  # Mostrar vectores de método YOLO
        
        # ===== SISTEMA DE CAPTURAS AUTOMÁTICAS =====
        self.SISTEMA_CAPTURAS_AUTO = 'yolo'  # Sistema usado para capturas: 'yolo' o 'supervivencia'
        
        # ===== EXPORTACIÓN DE DATOS =====
        self.OUTPUT_JSON_YOLO = "odometria_yolo.json"  # Archivo de salida para odometría YOLO
        self.OUTPUT_JSON_SUPERVIVENCIA = "odometria_supervivencia.json"  # Archivo de salida para odometría supervivencia

