import cv2
import numpy as np
from config import Q_ACT_BASE, Y_TOLERANCE, MIN_DISPARITY, MAX_DISPARITY # ¡CORREGIDO!

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


def get_cns(cns_filt, q_w, q_h, w):

    Q_ACT_DRAW = Q_ACT_BASE
    m_x = w // 2

    cns_all = []
    cns, h = cv2.findContours(cns_filt, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if cns:
        for i in range(len(cns)):
            if h[0][i][3] != -1 and cv2.contourArea(cns[i]) >= 50:
                M = cv2.moments(cns[i])
                if M["m00"] != 0:
                    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

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
