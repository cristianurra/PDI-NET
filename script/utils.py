import math
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any, Optional, Generator

from config import ConfiguracionGlobal

def dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Calcula la distancia entre dos puntos en 2D.
    
    con d = sqrt((x2-x1)² + (y2-y1)²)
    
    Args:
        p1: Primer punto como tupla (x, y)
        p2: Segundo punto como tupla (x, y)
    
    Returns:
        Distancia euclidiana entre los dos puntos
    """
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def depth_to_color(depth_cm: float, config: ConfiguracionGlobal) -> Tuple[int, int, int]:
    """
    Convierte una medida de profundidad en centímetros a un color BGR para visualización.
    
    Mapeo de colores según profundidad:
    - Objetos cercanos (MIN_DEPTH_CM): Rojo (255, 0, 0)
    - Objetos lejanos (MAX_DEPTH_CM): Azul (0, 0, 255)
    - Profundidades intermedias: Gradiente de rojo a azul
    - Fuera de rango: Azul puro
    
    Args:
        depth_cm: Profundidad en centímetros
        config: Objeto de configuración con MIN_DEPTH_CM y MAX_DEPTH_CM
    
    Returns:
        Tupla (B, G, R)
        """
    if depth_cm <= 0 or depth_cm >= config.MAX_DEPTH_CM:
        return (255, 0, 0)

    normalized_depth = np.clip(
        (depth_cm - config.MIN_DEPTH_CM) / (config.MAX_DEPTH_CM - config.MIN_DEPTH_CM),
        0.0,
        1.0
    )

    R = int(255 * (1 - normalized_depth))
    B = int(255 * normalized_depth)
    G = 0

    return (B, G, R)

def normalize_cell_view(current_image: np.ndarray, cell_target_size: Tuple[int, int] = (100, 100)) -> np.ndarray:
    """
    Redimensiona una imagen a un tamaño específico para visualización normalizada.
    
    Utiliza interpolación INTER_AREA,
    
    produciendo suavizados y evitando artefactos de muestreo.
    
    Args:
        current_image: Imagen de entrada (numpy array)
        cell_target_size: Tamaño objetivo como tupla (ancho, alto). Por defecto (100, 100)
    
    Returns:
        Imagen redimensionada al tamaño objetivo, o la imagen original si falla
    
    Note:
        Si ocurre algún error durante el redimensionamiento (ej. imagen inválida),
        retorna la imagen original sin modificar.
    """
    try:
        normalized_image = cv2.resize(current_image, cell_target_size, interpolation=cv2.INTER_AREA)
        return normalized_image
    except Exception:
        return current_image

def register_image_to_map(current_image: np.ndarray, existing_image: np.ndarray) -> np.ndarray:
    """
    Fusiona dos imágenes mediante promediado ponderado para registro y acumulación.
    
    Combina una imagen existente con una nueva imagen usando pesos iguales (50%-50%).
    Útil para crear mapas acumulativos o superponer información de múltiples frames.
    
    resultado = existing_image * 0.5 + current_image * 0.5
    
    Args:
        current_image: Nueva imagen a fusionar
        existing_image: Imagen existente/acumulada
    
    Returns:
        Imagen fusionada si es posible, caso contrario:
        - current_image si existing_image es None o tienen diferentes dimensiones
        - existing_image si ocurre un error durante la fusión
    
    Note:
        Ambas imágenes deben tener las mismas dimensiones (alto, ancho, canales)
        para poder realizar la fusión.
    """
    if existing_image is None or current_image is None or existing_image.shape != current_image.shape:
        return current_image

    try:
        fused_image = cv2.addWeighted(existing_image, 0.5, current_image, 0.5, 0)
        return fused_image

    except Exception:
        return existing_image

def map_trans(hist_m: List[Tuple[float, float]], m_w: int, m_h: int, config: ConfiguracionGlobal) -> Tuple[float, float, float]:
    """
    Calcula los parámetros de transformación para visualizar un mapa 2D adaptativo.
    
    Determina la escala y desplazamientos necesarios para que todos los puntos
    del historial quepan dentro del área de visualización del mapa, manteniendo
    un padding y centrando el contenido.
    
    Proceso:
    1. Encuentra el bounding box de todos los puntos históricos
    2. Calcula la escala para que quepan con padding (MAP_PAD_PX)
    3. Limita la escala máxima a MAP_ESC_V
    4. Centra el contenido en el mapa
    
    Args:
        hist_m: Lista de puntos históricos como tuplas (x, y) en coordenadas del mundo
        m_w: Ancho del mapa en píxeles
        m_h: Alto del mapa en píxeles
        config: Configuración global con MAP_PAD_PX y MAP_ESC_V
    
    Returns:
        Tupla (escala, offset_x, offset_y) donde:
        - escala: Factor de escala píxeles/unidad-mundo
        - offset_x: Desplazamiento horizontal para centrar
        - offset_y: Desplazamiento vertical para centrar
    
    Note:
        Si hist_m está vacío o todos los puntos son iguales, usa valores por defecto
        centrados con escala MAP_ESC_V.
    """
    if not hist_m:
        return config.MAP_ESC_V, m_w / 2, m_h / 2

    xs, ys = [p[0] for p in hist_m], [p[1] for p in hist_m]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    r_x, r_y = max_x - min_x, max_y - min_y

    sz_disp = m_w - 2 * config.MAP_PAD_PX
    r_max = max(r_x, r_y)

    esc_m = sz_disp / r_max if r_max > 0 else config.MAP_ESC_V
    esc_m = min(esc_m, config.MAP_ESC_V)

    c_x, c_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    off_x = m_w / 2 - (c_x * esc_m)
    off_y = m_h / 2 + (c_y * esc_m)

    if r_x == 0 and r_y == 0:
        off_x, off_y = m_w / 2, m_h / 2
        esc_m = config.MAP_ESC_V

    return esc_m, off_x, off_y

def open_svo_file(svo_path: str) -> Tuple[Optional[Generator[Tuple[bool, Optional[np.ndarray]], None, None]], int, int, int]:
    """
    Abre un archivo SVO (ZED stereo video) y crea un generador para leer frames.
    
    Los archivos SVO son grabaciones de la cámara estéreo ZED. Esta función:
    1. Inicializa la cámara ZED en modo reproducción (no tiempo real)
    2. Obtiene información del video (total frames, resolución)
    3. Crea un generador que produce frames estéreo concatenados horizontalmente
    
    Estructura del frame estéreo:
    [Imagen Izquierda | Imagen Derecha] - Concatenadas horizontalmente
    
    Args:
        svo_path: Ruta al archivo .svo
    
    Returns:
        Tupla con:
        - Generador que produce (success, frame) en cada iteración:
            * success: True si se leyó correctamente, False al finalizar
            * frame: numpy array BGR con ambas imágenes concatenadas, o None
        - total_frames: Número total de frames en el video
        - w: Ancho total del frame estéreo (ancho_izq + ancho_der)
        - h: Alto del frame
        
        Retorna (None, 0, 0, 0) si:
        - No está instalado pyzed
        - No se puede abrir el archivo SVO
    
    Dividir la imagen estéreo:
        >>> gen, total, width, height = open_svo_file("video.svo")
        >>> if gen:
        >>>     for success, frame in gen:
        >>>         if success:
        >>>             # Procesar frame estéreo
        >>>             left = frame[:, :width//2]
        >>>             right = frame[:, width//2:]
    
    Note:
        Requiere la biblioteca pyzed instalada (ZED SDK de Stereolabs).
        La cámara se cierra automáticamente cuando el generador termina.
    """
    try:
        import pyzed.sl as sl
    except ImportError:
        return None, 0, 0, 0

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        return None, 0, 0, 0

    total_frames = zed.get_svo_number_of_frames()
    cam_info = zed.get_camera_information()
    w = cam_info.camera_resolution.width * 2
    h = cam_info.camera_resolution.height

    def frame_generator():
        left_image = sl.Mat()
        right_image = sl.Mat()

        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                zed.retrieve_image(right_image, sl.VIEW.RIGHT)

                left_bgr = left_image.get_data()[:, :, :3]
                right_bgr = right_image.get_data()[:, :, :3]

                stereo_frame = np.hstack((left_bgr, right_bgr))

                yield True, stereo_frame
            else:
                yield False, None
                zed.close()
                break

    return frame_generator(), total_frames, w, h
