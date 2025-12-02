import cv2
import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional

from config import ConfiguracionGlobal

# Variable global para procesador CUDA (se inyecta desde gui)
_cuda_processor = None

def set_cuda_processor(cuda_proc):
    """Establece el procesador CUDA para usar en este módulo."""
    global _cuda_processor
    _cuda_processor = cuda_proc

def proc_seg(frame: np.ndarray, k_uni: np.ndarray, k_limp: np.ndarray) -> np.ndarray:
    """
    Procesa un frame para generar una imagen binaria segmentada con contornos de objetos.
    
    Pipeline de procesamiento:
    1. Conversión a escala de grises
    2. Suavizado gaussiano (3x3) para reducir ruido
    3. Umbralización adaptativa Gaussiana (inversa, blockSize=15, C=1)
    4. Cierre morfológico con k_uni para unir regiones cercanas
    5. Apertura morfológica con k_limp para limpiar ruido pequeño
    
    Args:
        frame: Imagen BGR de entrada
        k_uni: Kernel para unificar/consolidar regiones (operación de cierre)
        k_limp: Kernel para limpiar ruido (operación de apertura)
    
    Returns:
        Imagen binaria filtrada (255=objeto, 0=fondo) con contornos limpios
    
    Note:
        La umbralización adaptativa es más robusta a cambios de iluminación
        que el umbral global, usando ventanas locales de 15x15 píxeles.
    """
    # Usar CUDA si está disponible (solo si OpenCV tiene soporte CUDA)
    if _cuda_processor and _cuda_processor.opencv_cuda_available:
        gray = _cuda_processor.cvt_color_cuda(frame, cv2.COLOR_BGR2GRAY)
        blurred = _cuda_processor.gaussian_blur_cuda(gray, (3, 3), 0)
        binary = _cuda_processor.adaptive_threshold_cuda(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 1
        )
        closed = _cuda_processor.morphology_cuda(binary, cv2.MORPH_CLOSE, k_uni)
        filtered = _cuda_processor.morphology_cuda(closed, cv2.MORPH_OPEN, k_limp)
    else:
        # Procesamiento CPU tradicional (más común)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            src=blurred, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV, blockSize=15, C=1
        )
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_uni)
        filtered = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_limp)
    
    return filtered


def get_cns(cns_filt: np.ndarray, q_w: int, q_h: int, w: int, config: ConfiguracionGlobal, y_max_track: int = 9999) -> Tuple[List[Tuple[int, int]], List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], int]], List[Tuple[Tuple[int, int], int]]]:
    """
    Detecta contornos en imagen estéreo y realiza matching entre vistas izquierda/derecha.
    
    Proceso de matching estéreo:
    1. Encuentra todos los contornos válidos (área >= 50, en cuadrantes activos)
    2. Separa contornos en vista izquierda y derecha (mitad del frame)
    3. Empareja contornos L-R que cumplan:
       - Diferencia vertical <= Y_TOLERANCE (misma fila epipolar)
       - Disparidad dentro de [MIN_DISPARITY, MAX_DISPARITY]
    4. Calcula disparidad: disparity = x_left - x_right_ajustada
    
    Args:
        cns_filt: Imagen binaria filtrada con contornos
        q_w: Ancho de cada cuadrante de la grilla
        q_h: Alto de cada cuadrante de la grilla
        w: Ancho total del frame estéreo (izq + der)
        config: Configuración con Q_ACT_BASE, tolerancias y límites de disparidad
        y_max_track: Límite vertical máximo para tracking (no usado actualmente)
    
    Returns:
        Tupla con tres listas:
        1. cns_L_matched_only: Centroides izquierdos que tuvieron match [(x, y), ...]
        2. matched_cns_pairs: Pares completos [((cL, cR), disparity), ...]
        3. cns_disp_only: Centroides izq con su disparidad [((x, y), disp), ...]
    
    Note:
        Solo considera contornos internos (holes) mediante jerarquía [3] != -1.
        El matching es greedy: primer match válido se acepta.
    """
    m_x = w // 2

    cns_all = []
    contours, hier = cv2.findContours(cns_filt, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        for i in range(len(contours)):
            if hier is not None and hier[0][i][3] != -1 and cv2.contourArea(contours[i]) >= 50:
                M = cv2.moments(contours[i])
                if M["m00"] != 0:
                    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                    q_col, q_row = cX // q_w, cY // q_h
                    if (q_row, q_col) in config.Q_ACT_BASE:
                        cns_all.append((cX, cY))

    cns_L = [c for c in cns_all if c[0] < m_x]
    cns_R = [c for c in cns_all if c[0] >= m_x]

    matched_cns_pairs = []
    used_R = [False] * len(cns_R)

    for cL in cns_L:
        xL, yL = cL
        mejor_match_i = -1

        for i, cR in enumerate(cns_R):
            if not used_R[i]:
                xR, yR = cR
                xR_adj = xR - m_x

                if abs(yL - yR) <= config.Y_TOLERANCE:
                    disparity = xL - xR_adj

                    if config.MIN_DISPARITY <= disparity <= config.MAX_DISPARITY:
                        mejor_match_i = i
                        break

        if mejor_match_i != -1:
            cR_matched = cns_R[mejor_match_i]
            used_R[mejor_match_i] = True
            xR_adj = cR_matched[0] - m_x
            disparity = xL - xR_adj
            matched_cns_pairs.append(((cL, cR_matched), disparity))

    cns_L_matched_only = [pair[0][0] for pair in matched_cns_pairs]
    cns_disp_only = [(pair[0][0], pair[1]) for pair in matched_cns_pairs]

    return cns_L_matched_only, matched_cns_pairs, cns_disp_only


def proc_mesh_mask(frame: np.ndarray, mesh_consolidate_k: int, k_limp: np.ndarray, k_vert_fill: np.ndarray) -> Optional[np.ndarray]:
    """
    Genera una máscara binaria para detectar estructuras de malla/red en la imagen.
    
    Utiliza el canal de SATURACIÓN (HSV) en lugar de intensidad para mejor detección
    de texturas y patrones de malla independientes de iluminación.
    
    Pipeline de procesamiento:
    1. Conversión BGR -> HSV, extracción del canal de saturación
    2. Suavizado gaussiano (3x3)
    3. Umbralización adaptativa inversa (blockSize=15, C=1)
    4. Erosión + Dilatación con kernel cuadrado (apertura modificada)
    5. Cierre vertical con kernel rectangular (3 x 31) para conectar líneas verticales
    6. Apertura final con k_limp para limpieza
    
    Args:
        frame: Imagen BGR de entrada
        mesh_consolidate_k: Tamaño del kernel cuadrado para consolidar (típ. 7)
        k_limp: Kernel pequeño para limpieza final (típ. 3x3)
        k_vert_fill: Kernel vertical para rellenar gaps en estructuras verticales
    
    Returns:
        Máscara binaria con la malla detectada, o None si frame es None
    
    Note:
        El kernel vertical (3x31) es clave para detectar patrones de malla con
        líneas verticales dominantes. Configuración típica:
        - mesh_consolidate_k = 7 (MESH_CONSOLIDATE_K)
        - k_limp = 3x3 (K_LIMP)
        - k_vert_fill = 3x31 (K_VERT_FILL)
    """
    if frame is None:
        return None

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    blurred_s = cv2.GaussianBlur(saturation, (3, 3), 0)

    binary_s = cv2.adaptiveThreshold(
        src=blurred_s, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV, blockSize=15, C=1
    )

    binary = binary_s

    kernel_sq = np.ones((mesh_consolidate_k, mesh_consolidate_k), np.uint8)
    eroded = cv2.erode(binary, kernel_sq, iterations=1)
    opened = cv2.dilate(eroded, kernel_sq, iterations=1)

    closed_vertical = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_vert_fill, iterations=1)

    filtered = cv2.morphologyEx(closed_vertical, cv2.MORPH_OPEN, k_limp, iterations=1)

    return filtered


def get_mesh_boundary_y_pos(consolidated_mask: Optional[np.ndarray], w_half: int, h_total: int, kernel_edge: np.ndarray) -> int:
    """
    Detecta la posición Y del borde superior de la malla en la vista izquierda.
    
    Algoritmo de detección de borde:
    1. Extrae mitad izquierda de la máscara consolidada
    2. Binariza para asegurar valores 0/255
    3. Detecta borde restando: boundary = original - erosionada
    4. Encuentra contornos en el borde detectado
    5. Filtra contornos grandes (área > 100) en mitad superior de la imagen
    6. Retorna la mediana de las posiciones Y mínimas válidas
    
    Args:
        consolidated_mask: Máscara binaria de la malla (frame completo estéreo)
        w_half: Ancho de media imagen (para separar izq/der)
        h_total: Alto total del frame
        kernel_edge: Kernel para erosión en detección de bordes
    
    Returns:
        Posición Y (fila) del borde superior de la malla, o 0 si no se detecta
    
    Note:
        Solo considera la ROI superior (y < h_total/2) para evitar falsos positivos
        en la parte inferior. Usa mediana para robustez ante outliers.
    """
    if consolidated_mask is None:
        return 0

    mask_left = consolidated_mask[:, :w_half].copy()
    if mask_left.dtype != np.uint8 or len(mask_left.shape) != 2:
        mask_left = cv2.convertScaleAbs(mask_left)

    _, mask_left = cv2.threshold(mask_left, 1, 255, cv2.THRESH_BINARY)

    eroded = cv2.erode(mask_left, kernel_edge, iterations=1)
    boundary_mask = cv2.subtract(mask_left, eroded)

    contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_y_positions = []
    roi_top_y = h_total // 2

    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            if area > 100:
                current_min_y = int(np.min(c[:, :, 1]))
                if current_min_y < roi_top_y:
                    valid_y_positions.append(current_min_y)

    if valid_y_positions:
        return int(np.median(valid_y_positions))

    return 0


def detect_orange_markers(bgr_image: np.ndarray, config: ConfiguracionGlobal) -> List[Dict[str, Any]]:
    """
    Detecta marcadores naranjas circulares en la imagen usando segmentación por color.
    
    Pipeline de detección:
    1. Conversión BGR -> HSV para segmentación robusta por color
    2. Umbralización por rango HSV [ORANGE_HSV_LOW, ORANGE_HSV_HIGH]
    3. Apertura morfológica (3x3) para eliminar ruido pequeño
    4. Cierre morfológico (3x3) para rellenar huecos
    5. Detección de contornos externos
    6. Filtrado por criterios:
       - Área: [ORANGE_MIN_AREA, ORANGE_MAX_AREA]
       - Circularidad: >= ORANGE_CIRCULARITY
    
    Circularidad = 4π × área / perímetro²
    - Círculo perfecto: 1.0
    - Valores típicos aceptables: >= 0.4
    
    Args:
        bgr_image: Imagen BGR de entrada
        config: Configuración con rangos HSV y umbrales de validación
    
    Returns:
        Lista de diccionarios, cada uno con:
        - 'cx': Coordenada X del centroide
        - 'cy': Coordenada Y del centroide  
        - 'area': Área del marcador en píxeles
        - 'circ': Circularidad calculada (0-1)
        - 'bbox': Bounding box como tupla (x, y, ancho, alto)
    
    Note:
        Configuración típica (config.py):
        - ORANGE_HSV_LOW = (5, 120, 150)
        - ORANGE_HSV_HIGH = (22, 255, 255)
        - ORANGE_MIN_AREA = 30
        - ORANGE_MAX_AREA = 5000
        - ORANGE_CIRCULARITY = 0.4
    """
    if bgr_image is None:
        return []

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lower = np.array(config.ORANGE_HSV_LOW, dtype=np.uint8)
    upper = np.array(config.ORANGE_HSV_HIGH, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < config.ORANGE_MIN_AREA or area > config.ORANGE_MAX_AREA:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue
        circ = 4.0 * math.pi * area / (perimeter * perimeter)

        if circ < config.ORANGE_CIRCULARITY:
            continue

        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        x, y, w_box, h_box = cv2.boundingRect(c)

        results.append({'cx': cX[0], 'cy': cX[1], 'area': area, 'circ': circ, 'bbox': (x, y, w_box, h_box)})

    return results
