import numpy as np
from scipy.linalg import null_space

def coord_to_homogeneous_coord(u: np.ndarray) -> np.ndarray:
    """
    converting normal 2D coordinates to 3D homogeneous coordinates
    e.g. [[x1, y1], [x2, y2]] -> [[x1, y1, 1], [x2, y2, 1]]
    :param u: N-by-2 coordinates
    :return: N-by-3 homogeneous coordinates
    """
    if u.shape[1] != 2:
        print('u should be in size (N, 2)')
        return None

    return np.insert(u, 2, 1, axis=1)

def homogeneous_coord_to_coord(u: np.ndarray) -> np.ndarray:
    """
    converting 3D homogeneous coordinates to normal 2D coordinates
    e.g. [[x1, y1, 1], [x2, y2, 1]] -> [[x1, y1], [x2, y2]]
    :param u: N-by-3 homogeneous coordinates
    :return: N-by-2 coordinates
    """
    if u.shape[1] != 3:
        print('u should be in size (N, 3)')
        return None

    return np.delete(u, 2, axis=1)

def transform_with_H(H: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    :param H: 3x3 H matrix
    :param u: Nx3 src homogeneous coordinates
    :return:  Nx3 dst homogeneous coordinates
    """
    if H.shape != (3, 3):
        print(f'H shape: {H.shape} != 3x3!!')
        return None
    if u.shape[1] != 3:
        print(f'u shape: {u.shape} != Nx3!!')
        return None
    new_coord = np.dot(H, u.T).T

    scalar = (1/new_coord[:, -1]).reshape(-1, 1) # norm w to 1
    new_coord = new_coord*scalar

    return new_coord.astype(int)

def form_A(u: list, v: list) -> np.ndarray:
    """
    forming A for function solve_homography(u, v)
    [
        0   0   0  -x  -y  -1  xy'   yy'  y'
        x   y   1   0   0   0  -xx' -yx' -x'
    ]
    :param u: 1-by-2 source pixel location
    :param v: 1-by-2 source pixel location
    :return: 2-by-9 matrix
    """

    first_row  = np.array([0, 0, 0, -u[0], -u[1], -1, u[0]*v[1], u[1]*v[1], v[1]])
    second_row = np.array([u[0], u[1], 1, 0, 0, 0, -u[0]*v[0], -u[1]*v[0], -v[0]])
    sub_A = np.stack((first_row, second_row), axis=1)
    return sub_A

def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
        return None

    # TODO: 1.forming A
    A = form_A(u[0], v[0])
    for i in range(1, N):
        sub_A = form_A(u[i], v[i])
        A = np.concatenate((A, sub_A), axis=1)
        # print(f'A.shape: {A.shape}')
    
    # TODO: 2.solve H with A
    U, sigma, V = np.linalg.svd(A.T)
    H = V[-1].reshape(3, 3)

    # ns = null_space(A.T)[:, 0]
    # H = ns.reshape(3, 3)
    return H

def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    xv, yv = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
    src_coords = np.dstack([xv, yv]).reshape(-1, 2) # for forward warping
    dst_coords = np.dstack([xv, yv]).reshape(-1, 2) # for backward warping

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate

    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        dst_h_coords = coord_to_homogeneous_coord(dst_coords)

        src_h_coords = transform_with_H(H_inv, dst_h_coords)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)
        mask = (src_h_coords[:, 0] >= 0) & (src_h_coords[:, 0] < w_src) & (src_h_coords[:, 1] >= 0) & (src_h_coords[:, 1] < h_src)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates
        dst_coords   = dst_coords[mask]
        src_h_coords = src_h_coords[mask]

        # TODO: 6. assign to destination image with proper masking
        dst[dst_coords[:, 1], dst_coords[:, 0]] = src[src_h_coords[:, 1], src_h_coords[:, 0]]

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        src_h_coords = coord_to_homogeneous_coord(src_coords)

        dst_h_coords = transform_with_H(H, src_h_coords)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = (dst_h_coords[:, 0] >= 0) & (dst_h_coords[:, 0] < w_dst) & (dst_h_coords[:, 1] >= 0) & (dst_h_coords[:, 1] < h_dst)

        # TODO: 5.filter the valid coordinates using previous obtained mask
        src_coords = src_coords[mask]
        dst_h_coords = dst_h_coords[mask]

        # TODO: 6. assign to destination image using advanced array indicing
        dst[dst_h_coords[:, 1], dst_h_coords[:, 0]] = src[src_coords[:, 1], src_coords[:, 0]]


    return dst
