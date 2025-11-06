import cv2
import numpy as np
from config import Q_ACT_BASE, Y_TOLERANCE, MIN_DISPARITY, MAX_DISPARITY 

def proc_seg(frame, k_uni, k_limp):
    """
    Procesa un frame de video para segmentar objetos.
    
    Args: 
        frame (numpy.ndarray): Frame de video en formato BGR.
        k_uni (numpy.ndarray): Kernel para la operación de cierre morfológico.
        k_limp (numpy.ndarray): Kernel para la operación de apertura morfológica.
    
    Returns:
        numpy.ndarray: Máscara binaria segmentada.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        src=blurred, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV, blockSize=15, C=1
    )

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_uni)
    filtered = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_limp)
    return filtered


def get_cns(cns_filt, q_w, q_h, w, y_max_track=9999):
    """
    Extrae y empareja centroides de objetos detectados en una imagen estéreo.
    
    Args:
        cns_filt (numpy.ndarray): Máscara binaria segmentada de toda la imagen estéreo (izquierda y derecha).
        q_w (int): Ancho de cada celda de la cuadrícula.
        q_h (int): Alto de cada celda de la cuadrícula.
        w (int): Ancho total de la imagen estéreo (ambos ojos).
        y_max_track (int, optional): Coordenada Y máxima para filtrar detecciones. 
                                      Solo se aceptan puntos con cY > y_max_track. Default: 9999.
    
    Returns:
        tuple: (cns_L_matched_only, matched_cns_pairs, cns_disp_only)
            - cns_L_matched_only (list): Lista de centroides del ojo izquierdo que fueron emparejados [(x, y), ...].
            - matched_cns_pairs (list): Lista de tuplas ((centroide_L, centroide_R), disparidad).
            - cns_disp_only (list): Lista de tuplas (centroide_L, disparidad) para centroides emparejados.
    """

    Q_ACT_DRAW = Q_ACT_BASE
    m_x = w // 2

    cns_all = []
    contours, hier = cv2.findContours(cns_filt, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # altura de la imagen (en píxeles) tomada desde la máscara de entrada
    img_h = cns_filt.shape[0] if hasattr(cns_filt, 'shape') else None

    # Si y_max_track está en modo fallback (por ejemplo 0) o no es válido respecto a la altura,
    # interpretamos que no hay recorte dinámico y aceptamos todos los puntos.
    # En caso contrario, aceptamos SOLO puntos cuya coordenada Y sea mayor a y_max_track
    # (es decir, puntos por debajo de la línea de tracking).
    if contours:
        for i in range(len(contours)):
            # jerarquía puede ser None; solo comprobamos si existe
            if hier is not None and hier[0][i][3] != -1 and cv2.contourArea(contours[i]) >= 50:
                M = cv2.moments(contours[i])
                if M["m00"] != 0:
                    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

                    # ACEPTACIÓN: aceptar el punto si su coordenada Y es mayor que y_max_track
                    # (es decir, está DEBAJO de la línea roja). Si y_max_track == 0 (fallback),
                    # la condición cY > 0 deja pasar casi todos los puntos.
                    try:
                        if cY > y_max_track:
                            q_col, q_row = cX // q_w, cY // q_h
                            if (q_row, q_col) in Q_ACT_DRAW:
                                cns_all.append((cX, cY))
                    except Exception:
                        # En caso de comparación insegura con tipos inesperados,
                        # fallback a aceptar el punto para evitar pérdida de datos.
                        q_col, q_row = cX // q_w, cY // q_h
                        if (q_row, q_col) in Q_ACT_DRAW:
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

                if abs(yL - yR) <= Y_TOLERANCE:
                    disparity = xL - xR_adj

                    if MIN_DISPARITY <= disparity <= MAX_DISPARITY:
                        mejor_match_i = i
                        break

        if mejor_match_i != -1:
            cR_matched = cns_R[mejor_match_i]
            used_R[mejor_match_i] = True
            xR_adj = cR_matched[0] - m_x
            disparity = xL - xR_adj
            matched_cns_pairs.append(((cL, cR_matched), disparity))

    cns_L_matched_only = [pair[0][0] for pair in matched_cns_pairs]

    return cns_L_matched_only, matched_cns_pairs, [(pair[0][0], pair[1]) for pair in matched_cns_pairs]


def get_consolidated_mesh_mask(binary_mask, consolidate_k_size=5, k_vert_fill=None):
    """
    Aplica Apertura y Cierre vertical para consolidar las líneas de la malla en regiones sólidas.
    
    Args:
        binary_mask (numpy.ndarray): Máscara binaria de entrada.
        consolidate_k_size (int, optional): Tamaño del kernel cuadrado para apertura. Default: 5.
        k_vert_fill (numpy.ndarray, optional): Kernel rectangular vertical para cierre. 
                                                Si es None, se crea uno de 3x31. Default: None.
    
    Returns:
        numpy.ndarray: Máscara consolidada con operaciones morfológicas aplicadas, o None si la entrada es None.
    """
    if binary_mask is None:
        return None

    # Kernel cuadrado ajustable
    kernel_sq = np.ones((consolidate_k_size, consolidate_k_size), np.uint8)

    # 1. Apertura: Erosiona (filtra ruido) y luego Dilata (restaura forma)
    eroded = cv2.erode(binary_mask, kernel_sq, iterations=1)
    opened = cv2.dilate(eroded, kernel_sq, iterations=1)

    # 2. Cierre Vertical: Rellena las bandas horizontales negras (reflejos/huecos)
    if k_vert_fill is None:
        k_vert_fill = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 31))

    closed_vertical = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_vert_fill, iterations=1)

    return closed_vertical


def proc_mesh_mask(frame, mesh_consolidate_k, k_limp, k_vert_fill):
    """
    Genera una máscara binaria optimizada para consolidar la región de la malla enrollada.
    
    Utiliza el canal de saturación (HSV) para detectar la malla y aplica operaciones morfológicas
    (Apertura + Cierre vertical) para consolidar líneas en regiones sólidas.
    
    Args:
        frame (numpy.ndarray): Frame de video en formato BGR.
        mesh_consolidate_k (int): Tamaño del kernel cuadrado para operaciones de apertura.
        k_limp (numpy.ndarray): Kernel pequeño para limpieza final.
        k_vert_fill (numpy.ndarray): Kernel rectangular vertical para cierre vertical.
    
    Returns:
        numpy.ndarray: Máscara binaria consolidada de la malla, o None si frame es None.
    """
    if frame is None:
        return None

    # 1. Convertir a HLS y obtener Luminosidad
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    lightness = hls[:, :, 1]
    blurred_l = cv2.GaussianBlur(lightness, (3, 3), 0)

    # 2. Binarización adaptativa sobre luminosidad (reflejo brillante)
    binary_l = cv2.adaptiveThreshold(
        src=blurred_l, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY, blockSize=15, C=1
    )

    # Nota: En muchas escenas el agua/reflejo es muy brillante (L alto) y aparece en binary_l.
    # Sin embargo, para obtener el hilo de la malla preferimos la máscara por Saturación.

    # Ruta por Saturación (más estable para la malla)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    blurred_s = cv2.GaussianBlur(saturation, (3, 3), 0)

    binary_s = cv2.adaptiveThreshold(
        src=blurred_s, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV, blockSize=15, C=1
    )

    # Por ahora usamos la máscara por saturación como base para no romper la lógica actual
    # (la detección de contornos espera que la malla sea blanca en la máscara).
    binary = binary_s

    # 3. Consolidación de Malla (Apertura + Cierre Vertical)
    kernel_sq = np.ones((mesh_consolidate_k, mesh_consolidate_k), np.uint8)
    eroded = cv2.erode(binary, kernel_sq, iterations=1)
    opened = cv2.dilate(eroded, kernel_sq, iterations=1)

    if k_vert_fill is None:
        k_vert_fill = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 31))

    closed_vertical = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_vert_fill, iterations=1)

    # 4. Limpieza final con kernel pequeño
    filtered = cv2.morphologyEx(closed_vertical, cv2.MORPH_OPEN, k_limp, iterations=1)

    return filtered


def get_mesh_boundary(consolidated_mask, w_half, h_total, kernel_edge):
    """
    Calcula el borde (contorno exterior) de la zona consolidada de la malla en el ojo izquierdo.
    
    Args:
        consolidated_mask (numpy.ndarray): Máscara consolidada de la región de la malla.
        w_half (int): Ancho medio de la imagen (para separar ojo izquierdo del derecho).
        h_total (int): Altura total de la imagen.
        kernel_edge (numpy.ndarray): Kernel para erosión (obtención del borde fino).
    
    Returns:
        numpy.ndarray: Contorno más grande del borde de la malla en la mitad superior, 
                       o None si no se encuentra ningún contorno válido.
    """
    if consolidated_mask is None:
        return None
        
    # 1. Aislar el ojo izquierdo
    mask_left = consolidated_mask[:, :w_half].copy()
    if mask_left.dtype != np.uint8 or len(mask_left.shape) != 2:
        mask_left = cv2.convertScaleAbs(mask_left)
        
    # 2. Binarización de la máscara consolidada si no lo es
    _, mask_left = cv2.threshold(mask_left, 1, 255, cv2.THRESH_BINARY)
    
    # 3. Erosión con kernel pequeño para obtener un borde fino
    eroded = cv2.erode(mask_left, kernel_edge, iterations=1)

    # 4. Borde: resta la original menos la erosionada
    boundary_mask = cv2.subtract(mask_left, eroded)

    # 5. Encontrar el contorno más grande en la mitad superior
    contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    main_boundary = None
    max_area = 0
    roi_top_y = h_total // 2 

    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            if area > 100: 
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cY = int(M["m01"] / M["m00"])
                    
                    # Filtra y selecciona el contorno más grande en la mitad superior
                    if cY < roi_top_y:
                        if area > max_area:
                            max_area = area
                            main_boundary = c

    return main_boundary


def get_mesh_boundary_y_pos(consolidated_mask, w_half, h_total, kernel_edge):
    """
    Calcula la coordenada Y promedio del borde superior de la malla consolidada.
    
    Utiliza un método de votación (mediana) sobre los puntos más altos de los contornos válidos
    para obtener una estimación robusta del borde superior.
    
    Args:
        consolidated_mask (numpy.ndarray): Máscara consolidada de la región de la malla.
        w_half (int): Ancho medio de la imagen (para separar ojo izquierdo del derecho).
        h_total (int): Altura total de la imagen.
        kernel_edge (numpy.ndarray): Kernel para erosión (obtención del borde fino).
    
    Returns:
        int: Coordenada Y de la posición del borde superior de la malla (mediana de los puntos más altos).
             Retorna 0 si no se detecta ningún borde válido (modo fallback).
    """
    if consolidated_mask is None:
        return 0

    mask_left = consolidated_mask[:, :w_half].copy()
    if mask_left.dtype != np.uint8 or len(mask_left.shape) != 2:
        mask_left = cv2.convertScaleAbs(mask_left)

    _, mask_left = cv2.threshold(mask_left, 1, 255, cv2.THRESH_BINARY)

    # Borde: A - (A erosionada)
    eroded = cv2.erode(mask_left, kernel_edge, iterations=1)
    boundary_mask = cv2.subtract(mask_left, eroded)

    contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_y_positions = []

    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            # Solo si es un contorno significativo (no ruido)
            if area > 100:
                current_min_y = int(np.min(c[:, :, 1]))
                # Descartar si el borde está muy abajo (ej. debajo de la mitad de la imagen)
                if current_min_y < h_total // 2:
                    valid_y_positions.append(current_min_y)

    # Método de Voto: mediana para resiliencia frente a outliers
    if valid_y_positions:
        return int(np.median(valid_y_positions))

    # FALLBACK: Si no se detecta ningún borde significativo en la mitad superior.
    return 0
