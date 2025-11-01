 
import numpy as np
import math
import random

NOM_VID = 'stereonr.mp4'
RAD_PUN = 6
UMB_DIST = 50
N_VEL_PR = 3
PORC_MOS = 1.0
MIN_SUPERVIVENCIA_FR = 10
Q_X = 6
Q_Y = 5
Q_ACT = [(1, 1), (1, 4), (2, 1), (2, 4), (3, 1), (3, 4)]
SEP_CM = 6.0
SEP_PX_EST = 10
CM_POR_PX = SEP_CM / SEP_PX_EST
ESC_VEC = 5
MAP_CAN_SZ = 400
MAP_DISP_SZ = 300
MAP_PAD_PX = 40
MAP_ESC_V = 5.0
C_ACT = (0, 255, 0)
C_GRIS = (100, 100, 100)
C_NARAN = (0, 165, 255)
C_VEC_LR = (255, 255, 255)
C_CAM = (0, 255, 255)
C_MAP_FND = (0, 0, 0)
C_MAP_TXT = (255, 255, 255)
C_MAP_TRA = (150, 150, 150)
C_MAP_ACT = (0, 0, 255)


class Tracker:
    def __init__(self, max_d, len_v):
        self.objs = []
        self.prox_id = 0
        self.max_d = max_d
        self.len_v = len_v

    def update_and_get(self, cns_nue):

        from utils import dist

        usado_n = [False] * len(cns_nue)
        objs_sobrev = []

        for obj in self.objs:
            mejor_i, min_d = -1, self.max_d

            for i, c_nue in enumerate(cns_nue):
                if not usado_n[i] and dist(obj['pos'], c_nue) < min_d:
                    min_d = dist(obj['pos'], c_nue)
                    mejor_i = i

            if mejor_i != -1:
                pos_nue = cns_nue[mejor_i]
                vel = (pos_nue[0] - obj['pos'][0], pos_nue[1] - obj['pos'][1])

                obj['pos'] = pos_nue
                obj['hist_vel'].append(vel)
                if len(obj['hist_vel']) > self.len_v:
                    obj['hist_vel'].pop(0)

                obj['supervivencia_fr'] += 1

                usado_n[mejor_i] = True
                objs_sobrev.append(obj)
            else:
                pass

        for i, c_nue in enumerate(cns_nue):
            if not usado_n[i]:
                objs_sobrev.append({
                    'id': self.prox_id,
                    'pos': c_nue,
                    'hist_vel': [(0, 0)],
                    'supervivencia_fr': 1,
                    'color': (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
                })
                self.prox_id += 1

        self.objs = objs_sobrev
        return self.objs
