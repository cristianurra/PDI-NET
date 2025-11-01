import math
from config import MAP_PAD_PX, MAP_ESC_V

def dist(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def map_trans(hist_m, m_w, m_h):
    if not hist_m:
        return MAP_ESC_V, m_w / 2, m_h / 2

    xs, ys = [p[0] for p in hist_m], [p[1] for p in hist_m]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    r_x, r_y = max_x - min_x, max_y - min_y

    sz_disp = m_w - 2 * MAP_PAD_PX
    r_max = max(r_x, r_y)

    esc_m = sz_disp / r_max if r_max > 0 else MAP_ESC_V
    esc_m = min(esc_m, MAP_ESC_V)

    c_x, c_y = (min_x + max_x) / 2, (min_y + max_y) / 2
    off_x = m_w / 2 - (c_x * esc_m)
    off_y = m_h / 2 + (c_y * esc_m)

    if r_x == 0 and r_y == 0:
        off_x, off_y = m_w / 2, m_h / 2
        esc_m = MAP_ESC_V

    return esc_m, off_x, off_y
