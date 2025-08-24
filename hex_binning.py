# hex_binning.py
import numpy as np

def pixel_to_axial(x, y, size):
    """
    参考 redblobgames 的 cube/axial 转换。
    size: hex 边长
    假设点坐标已归一并以(0,0)为中心，如需偏移可加上 origin。
    """
    q = (np.sqrt(3)/3 * x - 1.0/3 * y) / size
    r = (2.0/3 * y) / size
    return cube_round(q, -q-r, r)

def cube_round(x, y, z):
    rx, ry, rz = round(x), round(y), round(z)
    dx, dy, dz = abs(rx-x), abs(ry-y), abs(rz-z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    # axial(q,r) = (x,z)
    return int(rx), int(rz)
