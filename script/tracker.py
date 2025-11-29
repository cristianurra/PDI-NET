import numpy as np
from typing import List, Tuple, Dict, Any

from config import ConfiguracionGlobal
from utils import dist, depth_to_color

class Tracker:
    def __init__(self, max_d: float, len_v: int, config: ConfiguracionGlobal):
        self.objs: List[Dict[str, Any]] = []
        self.prox_id = 0
        self.max_d = max_d
        self.len_v = len_v
        self.config = config

    def update_config(self, max_d: float, len_v: int):
        self.max_d = max_d
        self.len_v = len_v

    def update_and_get(self, matched_cns_pairs: List[Tuple[Tuple[Tuple[int, int], Tuple[int, int]], int]]) -> List[Dict[str, Any]]:
        max_d_act = self.max_d
        n_vel_pr_act = self.len_v

        cns_nue_L = [pair[0][0] for pair in matched_cns_pairs]
        pair_map = {pair[0][0]: (pair[0][1], pair[1]) for pair in matched_cns_pairs}
        usado_n = [False] * len(cns_nue_L)

        objs_sobrev_temp: List[Dict[str, Any]] = []

        for obj in self.objs:
            mejor_i, min_d = -1, max_d_act

            avg_vx = np.median([v[0] for v in obj.get('hist_vel', [(0, 0)])])
            avg_vy = np.median([v[1] for v in obj.get('hist_vel', [(0, 0)])])
            pos_pred = (obj['pos'][0] + avg_vx, obj['pos'][1] + avg_vy)

            for i, c_nue in enumerate(cns_nue_L):
                if not usado_n[i]:
                    d = dist(pos_pred, c_nue)
                    if d < min_d:
                        min_d = d
                        mejor_i = i

            if mejor_i != -1:
                pos_nue = cns_nue_L[mejor_i]
                pos_nue_R, disp = pair_map[pos_nue]
                vel = (pos_nue[0] - obj['pos'][0], pos_nue[1] - obj['pos'][1])

                hist_pos = obj.get('hist_pos', [])
                hist_pos.append(pos_nue)
                if len(hist_pos) > self.config.FRAMES_MAX_ESTATICO:
                    hist_pos.pop(0)
                obj['hist_pos'] = hist_pos

                is_static = len(hist_pos) == self.config.FRAMES_MAX_ESTATICO and all(p == pos_nue for p in hist_pos)

                if is_static:
                    obj['supervivencia_fr'] = 0
                else:
                    obj['pos'] = pos_nue
                    obj['pos_R'] = pos_nue_R

                    obj.setdefault('hist_vel', [(0, 0)]).append(vel)
                    if len(obj['hist_vel']) > n_vel_pr_act:
                        obj['hist_vel'].pop(0)

                    current_depth = (self.config.FOCAL_PIX * self.config.BASELINE_CM) / disp if disp > 1.0 else 0.0
                    obj.setdefault('hist_depth', []).append(current_depth)
                    if len(obj['hist_depth']) > self.config.N_DEPTH_PR:
                        obj['hist_depth'].pop(0)

                    if obj['hist_depth'] and current_depth > 0:
                        avg_depth = np.median([d for d in obj['hist_depth'] if d > 0])
                        obj['depth_cm'] = avg_depth
                        obj['color'] = depth_to_color(avg_depth, self.config)

                    obj['supervivencia_fr'] += 1
                    usado_n[mejor_i] = True
                    objs_sobrev_temp.append(obj)

            else:
                obj['supervivencia_fr'] = max(0, obj['supervivencia_fr'] - 1)
                if obj['supervivencia_fr'] >= 1:
                    objs_sobrev_temp.append(obj)

        for i, c_nue_L in enumerate(cns_nue_L):
            if not usado_n[i]:
                pos_R, disp = pair_map[c_nue_L]
                depth = (self.config.FOCAL_PIX * self.config.BASELINE_CM) / disp if disp > 1.0 else 0.0
                hist_depth_inicial = [depth] if depth > 0 else []
                color_inicial = depth_to_color(depth, self.config)

                objs_sobrev_temp.append({
                    'id': self.prox_id,
                    'pos': c_nue_L,
                    'pos_R': pos_R,
                    'hist_vel': [(0, 0)],
                    'supervivencia_fr': 1,
                    'color': color_inicial,
                    'depth_cm': depth,
                    'hist_depth': hist_depth_inicial,
                    'hist_pos': [c_nue_L]
                })
                self.prox_id += 1

        self.objs = [obj for obj in objs_sobrev_temp if obj['supervivencia_fr'] >= 1]

        return self.objs
