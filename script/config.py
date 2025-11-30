import numpy as np
import cv2

class ConfiguracionGlobal:
    def __init__(self, nom_vid=""):
        self.NOM_VID = nom_vid
        self.RAD_PUN = 6
        self.UMB_DIST = 75
        self.N_VEL_PR = 10
        self.MIN_SUPERVIVENCIA_FR = 1
        self.FRAMES_MAX_ESTATICO = 3
        self.Q_X = 6
        self.Q_Y = 5
        self.SEP_CM = 2.5
        self.SEP_PX_EST = 20
        self.CM_POR_PX = self.SEP_CM / self.SEP_PX_EST

        self.BASELINE_CM = 12.0
        self.FOCAL_PIX = 800.0
        self.MIN_DEPTH_CM = 20.0
        self.MAX_DEPTH_CM = 300.0
        self.N_DEPTH_PR = 5
        self.MIN_DISPARITY = 5
        self.MAX_DISPARITY = 150
        self.Y_TOLERANCE = 6

        self.ESC_VEC = 20
        self.C_CAM = (0, 255, 255)
        self.FIXED_GRID_SIZE_CM = 40.0
        self.RECT_SZ_CM_FALLBACK = 30.0
        self.RECT_MARGIN_CM = 5.0
        self.MAP_PAD_PX = 40
        self.MAP_ESC_V = 5.0
        self.C_MAP_TXT = (255, 255, 255)
        self.C_MAP_ACT = (0, 0, 255)
        self.C_NARAN = (0, 165, 255)
        self.C_GRIS = (100, 100, 100)
        self.C_ACT = (0, 255, 0)
        self.C_DANO = (0, 0, 255)

        self.MESH_CONSOLIDATE_K = 7
        self.Y_MASK_OFFSET = 100

        self.K_UNI_SIZE = 5
        self.K_LIMP_SIZE = 3
        self.K_VERT_FILL_H = 31
        self.K_VERT_FILL_W = 3

        self.K_UNI = np.ones((self.K_UNI_SIZE, self.K_UNI_SIZE), np.uint8)
        self.K_LIMP = np.ones((self.K_LIMP_SIZE, self.K_LIMP_SIZE), np.uint8)
        self.K_VERT_FILL = cv2.getStructuringElement(cv2.MORPH_RECT, (self.K_VERT_FILL_W, self.K_VERT_FILL_H))

        self.Q_ACT_BASE = [(1, 1), (1, 4), (2, 1), (2, 4), (3, 1), (3, 4)]
        self.PORC_MOS_INT = 100
        self.PORC_MOS = 1.0
        self.EDGE_POINT_RADIUS = 2

        self.ORANGE_HSV_LOW = (5, 120, 150)
        self.ORANGE_HSV_HIGH = (22, 255, 255)
        self.ORANGE_MIN_AREA = 30
        self.ORANGE_MAX_AREA = 5000
        self.ORANGE_CIRCULARITY = 0.4

        self.MAP_CAN_SZ = 800
        self.START_FRAME = 0

        self.PROFUNDIDAD_STEREO_ACTIVA = True
        self.SKIP_RATE = 1
        self.N_FRAMES_HISTORIAL = 5
        self.MAP_ZOOM_FACTOR = 10

        self.DMG_ALPHA = 0.1
        self.DMG_NUM_NB = 4
        self.DMG_FRAMES = 6
        self.DMG_THRESHOLD = 2.5
        self.DMG_DIST_TRACK = 20
