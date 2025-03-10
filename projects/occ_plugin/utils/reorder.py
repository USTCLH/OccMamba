import numpy as np
import torch

def default_order_index_within_range(max_x=128, max_y=128, max_z=10):
    indices = torch.arange(max_x*max_y*max_z)
    indices = indices.reshape(max_x, max_y, max_z).transpose(0,2).reshape(-1)
    return indices

def default2D_order_index_within_range(max_x=128, max_y=128, max_z=10):
    return torch.arange(max_x*max_y*max_z)

def hilbert2D(limit, n, x=0, y=0, order=None):
    if n == 1:
        if x >= limit[0] or y >= limit[1] or x < 0 or y < 0:
            return []
        else:
            return [(x, y)]
    
    if order is None:
        order = [(0, 0), (0, 1), (1, 1), (1, 0)]  # 二维希尔伯特曲线的基本顺序
    
    n //= 2
    points = []
    for dx, dy in order:
        new_x = x + dx * n
        new_y = y + dy * n
        points += hilbert2D(limit, n, new_x, new_y, order=order)

    return points

def H2HE_order_index_within_range(max_x=128, max_y=128, max_z=10, coor_order='xy', inverse=False):
    # Create a mapping from coor_order to coordinate increments
    index_map = {'x': 0, 'y': 1}
    if not inverse:
        base_order = [(0, 0), (0, 1), (1, 1), (1, 0)]
    else:
        base_order = [(0, 0), (0, -1), (-1, -1), (-1, 0)]

    desired_order_indices = [index_map[ch] for ch in coor_order]
    order = [(tuple(coord[idx] for idx in desired_order_indices)) for coord in base_order]

    # Find nearest powers of two
    max_side = max(max_x, max_y)
    cube_side = 1
    while cube_side < max_side:
        cube_side *= 2
    
    # Generate points2D with the specified order
    if not inverse:
        points2D = hilbert2D((max_x, max_y), cube_side, order=order)
    else:
        points2D = hilbert2D((max_x, max_y), cube_side, max_x-1, max_y-1, order=order)

    # Filter points2D to fit exactly within the dimensions
    point2D_list = [p for p in points2D if p[0] < max_x and p[1] < max_y]

    idx = 0
    map_xyz_to_idx = {}
    for x, y in point2D_list:
        for z in range(max_z):
            map_xyz_to_idx[(x, y, z)] = idx
            idx += 1

    indices = []
    for x in range(max_x):
        for y in range(max_y):
            for z in range(max_z):
                indices.append(map_xyz_to_idx[(x, y, z)])

    indices = torch.tensor(indices)
    indices = indices.argsort()

    return indices