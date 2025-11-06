import cv2
import numpy as np

NOM_VID = r'C:\Users\ivan_\OneDrive\Escritorio\USM\2025-2\PDI Real\Proyecto\Primer push mio\PDI-NET\videos_malla_piscina\2025_09_11\2025_09_11_12_47_39.svo'
RAD_PUN = 6
UMB_DIST = 75
N_VEL_PR = 5
PORC_MOS = 1.0
MIN_SUPERVIVENCIA_FR = 20
FRAMES_MAX_ESTATICO = 3
Q_X = 6
Q_Y = 5
Q_ACT_BASE = [(1, 1), (1, 4), (2, 1), (2, 4), (3, 1), (3, 4)]  # Original: solo 6 cuadrantes
#Q_ACT_BASE = [(row, col) for row in range(Q_Y) for col in range(Q_X)]  # Todos los cuadrantes activos
SEP_CM = 2.5
SEP_PX_EST = 20
CM_POR_PX = SEP_CM / SEP_PX_EST

BASELINE_CM = 12.0 # Según manual estereocámara ZED X son 12 cm 
FOCAL_PIX = 800.0
MIN_DEPTH_CM = 20.0
MAX_DEPTH_CM = 300.0
N_DEPTH_PR = 5
MIN_DISPARITY = 5
MAX_DISPARITY = 150
Y_TOLERANCE = 3

ESC_VEC = 5
MAP_CAN_SZ = 800
MAP_PAD_PX = 40
MAP_ESC_V = 5.0
C_ACT = (0, 255, 0)
C_GRIS = (100, 100, 100)
C_NARAN = (0, 165, 255)
C_VEC_LR = (255, 255, 255)
C_CAM = (0, 255, 255)
C_MAP_FND = (0, 0, 0)
C_MAP_TXT = (255, 255, 255)
C_MAP_ACT = (0, 0, 255)
C_VEC_PUNTO = (0, 255, 255)

FIXED_GRID_SIZE_CM = 40.0
RECT_SZ_CM_FALLBACK = 30.0
RECT_MARGIN_CM = 5.0

# Parámetros para eliminar frames borrosos al inicio
VAR_LAPLACIAN_THRESH = 100.0  # umbral de varianza del Laplaciano para considerar un frame nítido
MAX_INITIAL_SKIP_FRAMES = 60  # máximo de frames iniciales a saltar si son borrosos

ORB_DETECTOR = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
FLANN_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

K_UNI = np.ones((5, 5), np.uint8)
K_LIMP = np.ones((3, 3), np.uint8)
# Tamaño del kernel para consolidar la malla mediante apertura morfológica (erosión+dilatación)
MESH_CONSOLIDATE_K = 7
# Kernel vertical para rellenar discontinuidades horizontales (reflejos/huecos)
K_VERT_FILL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 31))
# Offset (px) para definir el límite de tracking por debajo del borde detectado
Y_MASK_OFFSET = 100

# Parámetros para marcar bordes (visualización)
EDGE_CANNY_LOW = 50
EDGE_CANNY_HIGH = 150
EDGE_MAX_POINTS = 600
EDGE_POINT_RADIUS = 2

# Parámetros para detectar puntos naranjas de referencia en la malla (HSV)
ORANGE_HSV_LOW = (5, 120, 150)   # H,S,V
ORANGE_HSV_HIGH = (22, 255, 255)

# Parámetros para detectar zonas muy blancas (concentración de blanco) en ojo izquierdo
WHITE_INTENSITY_THRESH = 200
WHITE_MORPH_K = 5

# Parámetros adicionales para detección robusta de marcadores naranjas
ORANGE_MIN_AREA = 30       # px
ORANGE_MAX_AREA = 5000     # px
ORANGE_CIRCULARITY = 0.4   # 0..1, 1 = perfect circle
EXPORT_ORANGE_CSV = False  # si True exporta las detecciones a CSV
ORANGE_CSV_PATH = r"orange_detections.csv"
