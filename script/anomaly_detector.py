import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from config import ConfiguracionGlobal

class DamageDetector:
    def __init__(self, config: ConfiguracionGlobal):
        self.config = config
        self.candidates_prev: List[List[Any]] = []
        self.next_id = 1
        self.kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    def detect(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        height, width = frame_bgr.shape[:2]
        frame_result = frame_bgr.copy()
        confirmed_damages = []

        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)
        l_channel, _, _ = cv2.split(lab)

        th = cv2.adaptiveThreshold(
            src=l_channel,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=15,
            C=0
        )
        ########################################################
        cv2.imshow("DEBUG 1: Threshold Crudo", th)
        kernel_limpieza = np.ones((3, 3), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel_limpieza)
        n_white = cv2.countNonZero(th)
        if n_white > (width * height) / 2:
             th = cv2.bitwise_not(th)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, 4)
        if num_labels < 2:
            return frame_result, []

        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        net_mask = np.zeros((height, width), dtype=np.uint8)
        net_mask[labels == largest_label] = 255

        net_mask = cv2.morphologyEx(net_mask, cv2.MORPH_CLOSE, self.kernel_morph, iterations=1)

        holes_mask = cv2.bitwise_not(net_mask)
######################################################################################
        cv2.imshow("DEBUG: Lo que ve el detector", holes_mask)
        cv2.rectangle(holes_mask, (0,0), (width, height), 0, 10) 
        
        num_holes, _, hole_stats, hole_centroids = cv2.connectedComponentsWithStats(holes_mask, 4)

        areas = []
        rects = []
        centroids_valid = []
        valid_indices = []

        for i in range(1, num_holes): # Empezamos en 1 (0 es la red/fondo)
            area = hole_stats[i, cv2.CC_STAT_AREA]
            x, y, w, h = hole_stats[i, :4]
            
            # Descartar si toca los bordes (aunque el rect negro ayuda, verificamos)
            if x <= 1 or y <= 1 or (x+w) >= width-1 or (y+h) >= height-1:
                continue
                
            areas.append(area)
            rects.append((x, y, w, h))
            centroids_valid.append(hole_centroids[i])
            valid_indices.append(i)

        if not areas:
            return frame_result, []

        areas_np = np.array(areas)
        mean_area = np.mean(areas_np)
        std_area = np.std(areas_np)

        z_factor = 3 * (1 - np.exp(-self.config.DMG_ALPHA * len(areas)))
        threshold_area = mean_area + z_factor * std_area

        candidates_current = [] # Estructura: [centroide, rect, frame_cnt, id, area, max_neighbor_area]

        for i, area in enumerate(areas):
            if area > threshold_area:
                curr_ct = centroids_valid[i]
                
                dists = np.linalg.norm(np.array(centroids_valid) - curr_ct, axis=1)
                
                sorted_indices = np.argsort(dists)
                
                neighbor_indices = sorted_indices[1 : self.config.DMG_NUM_NB + 1]
                
                if len(neighbor_indices) < self.config.DMG_NUM_NB:
                    continue
                    
                neighbor_areas = areas_np[neighbor_indices]
                max_neighbor_area = np.max(neighbor_areas)

                candidates_current.append([
                    curr_ct,        
                    rects[i],       
                    0,              
                    0,              
                    area,           
                    max_neighbor_area 
                ])


        final_candidates_next_frame = []

        for prev in self.candidates_prev:
            prev_ct = prev[0]
            best_match_idx = -1
            min_dist = float('inf')


            for idx, curr in enumerate(candidates_current):
                dist = np.linalg.norm(curr[0] - prev_ct)
                if dist < min_dist:
                    min_dist = dist
                    best_match_idx = idx


            
            dist_threshold = self.config.DMG_DIST_TRACK
            if prev[2] > self.config.DMG_FRAMES:
                 dist_threshold = 100 # Tracking más laxo si ya está confirmado

            if best_match_idx != -1 and min_dist < dist_threshold:
                match = candidates_current[best_match_idx]
                

                if match[4] > self.config.DMG_THRESHOLD * match[5]:
                    
                    frames_alive = prev[2] + 1
                    current_id = prev[3]


                    if frames_alive == self.config.DMG_FRAMES and current_id == 0:
                        current_id = self.next_id
                        self.next_id += 1

                    match[2] = frames_alive
                    match[3] = current_id
                    

                    
                    final_candidates_next_frame.append(match)

                    if frames_alive >= self.config.DMG_FRAMES:
                        self._draw_damage(frame_result, match)
                        confirmed_damages.append({
                            'id': current_id,
                            'rect': match[1],
                            'centroid': match[0],
                            'area': match[4]
                        })


        for curr in candidates_current:
            # Si frame count es 0, significa que no fue actualizado por un previo
            if curr[2] == 0:
                final_candidates_next_frame.append(curr)

        self.candidates_prev = final_candidates_next_frame
        
        return frame_result, confirmed_damages

    def _draw_damage(self, img, candidate_data):
        x, y, w, h = candidate_data[1]
        dmg_id = candidate_data[3]
        

        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), self.config.C_DANO, -1) # Relleno
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img) # Mezcla alfa
        

        cv2.rectangle(img, (x, y), (x + w, y + h), self.config.C_DANO, 2)
        cv2.putText(img, f"DMG #{dmg_id}", (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.C_DANO, 2)