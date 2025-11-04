import cv2
import numpy as np

NOM_VID = 'stereonr_h264.mp4'
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

ORB_DETECTOR = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
FLANN_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

K_UNI = np.ones((5, 5), np.uint8)
K_LIMP = np.ones((3, 3), np.uint8)
