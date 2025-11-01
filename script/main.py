 
import cv2
import numpy as np
from config import NOM_VID, UMB_DIST, N_VEL_PR, Q_X, Q_Y, CM_POR_PX, Tracker
from processing import proc_seg, get_cns
from drawing import dib_ayu, dib_mov, dib_map, show_v

def main():
    cap = cv2.VideoCapture(NOM_VID)
    INI_FR = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, INI_FR)
    ret, frame = cap.read()

    if not ret:
        print("Error: ")
        return
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_w, n_h = w // 2, h // 2
    q_w, q_h = w // Q_X, h // Q_Y
    k_uni = np.ones((5, 5), np.uint8)
    k_limp = np.ones((3, 3), np.uint8)
    tracker = Tracker(UMB_DIST, N_VEL_PR)
    pos_m_x, pos_m_y = 0.0, 0.0
    hist_pts = [(0.0, 0.0)]
    while ret:
        cns_filt = proc_seg(frame, k_uni, k_limp)
        cns_act = get_cns(cns_filt, q_w, q_h)
        objs = tracker.update_and_get(cns_act)
        dib_ayu(frame, w, h, q_w, q_h)
        del_p_x, del_p_y = dib_mov(frame, objs, w, h)
        del_c_x = del_p_x * CM_POR_PX
        del_c_y = -del_p_y * CM_POR_PX

        pos_m_x += del_c_x
        pos_m_y += del_c_y
        hist_pts.append((pos_m_x, pos_m_y))
        canv_m = dib_map(hist_pts, pos_m_x, pos_m_y)
        if show_v(frame, canv_m, n_w, n_h):
            break

        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
