 
import cv2
from config import Q_ACT

def proc_seg(frame, k_uni, k_limp):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        src=blurred, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV, blockSize=15, C=1
    )

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_uni)
    filtered = cv2.morphologyEx(closed, cv2.MORPH_OPEN, k_limp)
    return filtered

def get_cns(cns_filt, q_w, q_h):
    cns_act = []
    cns, h = cv2.findContours(cns_filt, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if cns:
        for i in range(len(cns)):
            if h[0][i][3] != -1 and cv2.contourArea(cns[i]) >= 50:
                M = cv2.moments(cns[i])
                if M["m00"] != 0:
                    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    q_col, q_row = cX // q_w, cY // q_h
                    if (q_row, q_col) in Q_ACT:
                        cns_act.append((cX, cY))
    return cns_act
