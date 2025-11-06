import math
import numpy as np
import cv2
from config import MIN_DEPTH_CM, MAX_DEPTH_CM, MAP_ESC_V, MAP_PAD_PX # ¡CORREGIDO!
from config import ORANGE_HSV_LOW, ORANGE_HSV_HIGH, ORANGE_MIN_AREA, ORANGE_MAX_AREA, ORANGE_CIRCULARITY

def dist(p1, p2):
    """
    Calcula la distancia euclidiana entre dos puntos 2D.
    
    Args:
        p1 (tuple): Primera coordenada (x, y).
        p2 (tuple): Segunda coordenada (x, y).
    
    Returns:
        float: Distancia euclidiana entre p1 y p2.
    """
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def depth_to_color(depth_cm):
    """
    Convierte un valor de profundidad en centímetros a un color RGB.
    
    Mapea profundidades cercanas a rojo y profundidades lejanas a azul de forma lineal.
    
    Args:
        depth_cm (float): Profundidad en centímetros.
    
    Returns:
        tuple: Color BGR (B, G, R) donde cercano=rojo (0, 0, 255) y lejano=azul (255, 0, 0).
    """
    if depth_cm <= 0 or depth_cm >= MAX_DEPTH_CM:
        return (255, 0, 0)

    normalized_depth = np.clip(
        (depth_cm - MIN_DEPTH_CM) / (MAX_DEPTH_CM - MIN_DEPTH_CM),
        0.0,
        1.0
    )

    R = int(255 * (1 - normalized_depth))
    B = int(255 * normalized_depth)
    G = 0

    return (B, G, R)

def map_trans(hist_m, m_w, m_h):
    """
    Calcula la transformación (escala y offsets) para renderizar el mapa de posiciones visitadas.
    
    Args:
        hist_m (list): Lista de coordenadas (x, y) visitadas en centímetros.
        m_w (int): Ancho del canvas del mapa en píxeles.
        m_h (int): Alto del canvas del mapa en píxeles.
    
    Returns:
        tuple: (esc_m, off_x, off_y)
            - esc_m (float): Escala en píxeles por centímetro para el mapa.
            - off_x (float): Offset en X para centrar el mapa.
            - off_y (float): Offset en Y para centrar el mapa.
    """
    if not hist_m:
        return MAP_ESC_V, m_w / 2, m_h / 2

    xs, ys = [p[0] for p in hist_m], [p[1] for p in hist_m]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    r_x, r_y = max_x - min_x, max_y - min_y

    sz_disp = m_w - 2 * MAP_PAD_PX
    r_max = max(r_x, r_y)

    esc_m = sz_disp / r_max if r_max > 0 else MAP_ESC_V
    esc_m = min(esc_m, MAP_ESC_V)

    c_x, c_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    off_x = m_w / 2 - (c_x * esc_m)
    off_y = m_h / 2 + (c_y * esc_m)

    if r_x == 0 and r_y == 0:
        off_x, off_y = m_w / 2, m_h / 2
        esc_m = MAP_ESC_V

    return esc_m, off_x, off_y

def normalize_cell_view(current_image, cell_target_size=(100, 100)):
    """
    Normaliza (redimensiona) una imagen a un tamaño objetivo para almacenamiento en el mapa.
    
    Args:
        current_image (numpy.ndarray): Imagen a normalizar.
        cell_target_size (tuple, optional): Tamaño objetivo (ancho, alto) en píxeles. Default: (100, 100).
    
    Returns:
        numpy.ndarray: Imagen redimensionada o imagen original si falla el redimensionamiento.
    """
    try:
        normalized_image = cv2.resize(current_image, cell_target_size, interpolation=cv2.INTER_AREA)
        return normalized_image
    except Exception:
        return current_image

def register_image_to_map(current_image, existing_image):
    """
    Fusiona una imagen actual con una imagen existente del mapa mediante promedio ponderado.
    
    Args:
        current_image (numpy.ndarray): Imagen actual a fusionar.
        existing_image (numpy.ndarray): Imagen existente en el mapa.
    
    Returns:
        numpy.ndarray: Imagen fusionada (promedio 50-50) o imagen existente si la fusión falla.
    """
    if existing_image is None or current_image is None or existing_image.shape != current_image.shape:
        return current_image

    try:
        fused_image = cv2.addWeighted(existing_image, 0.5, current_image, 0.5, 0)
        return fused_image

    except Exception:
        return existing_image


def detect_orange_markers(bgr_image):
    """
    Detecta marcadores naranjas (lazos de referencia) en una imagen BGR.
    
    Utiliza segmentación por color en espacio HSV y filtros de área y circularidad
    para identificar objetos naranjas circulares.
    
    Args:
        bgr_image (numpy.ndarray): Imagen en formato BGR.
    
    Returns:
        list: Lista de diccionarios con información de cada marcador detectado:
              [{'cx': int, 'cy': int, 'area': float, 'circ': float, 'bbox': (x, y, w, h)}]
              Las coordenadas son relativas a la imagen de entrada.
    """
    if bgr_image is None:
        return []

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lower = np.array(ORANGE_HSV_LOW, dtype=np.uint8)
    upper = np.array(ORANGE_HSV_HIGH, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # Morfología para quitar ruido
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < ORANGE_MIN_AREA or area > ORANGE_MAX_AREA:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circ = 4.0 * math.pi * area / (perimeter * perimeter)

        if circ < ORANGE_CIRCULARITY:
            # no es suficientemente circular
            continue

        M = cv2.moments(c)
        if M['m00'] == 0:
            continue
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])

        x, y, w_box, h_box = cv2.boundingRect(c)

        results.append({'cx': cX, 'cy': cY, 'area': area, 'circ': circ, 'bbox': (x, y, w_box, h_box)})

    return results
