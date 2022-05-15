from cProfile import label
import numpy as np
import cv2.ximgproc as xip
import cv2

def gen_binary_pattern(img: np.ndarray) -> np.ndarray:
    '''
        generate binary pattern for each pixel
    '''
    # FIXME change to kernel = 5
    binary_pattern = []
    padded_img = np.pad(img, ((3, 3), (3, 3), (0, 0)), 'constant', constant_values=0)
    for i in range(5):
        for j in range(5):
            pattern = padded_img[i+1:i-5, j+1:j-5, :] > padded_img[3:-3, 3:-3, :]
            binary_pattern.append(pattern)

    binary_pattern = np.stack(binary_pattern, axis=-1)
    return binary_pattern

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency

    Il_binary_pattern = gen_binary_pattern(Il) # shape = (h, w, ch, 25)
    Ir_binary_pattern = gen_binary_pattern(Ir) # shape = (h, w, ch, 25)
    Ir_2_Il_cost = [] # fix Il, shift Ir leftward
    Il_2_Ir_cost = [] # fix Ir, shift Il rightward

    # 0 case
    cost = np.count_nonzero(np.not_equal(Il_binary_pattern, Ir_binary_pattern), axis=-1) # shape = (h, w, ch)
    cost = np.sum(cost, axis=-1)
    Ir_2_Il_cost.append(cost)
    Il_2_Ir_cost.append(cost)

    for i in range(1, max_disp):
        # cost = hamming distance
        # shift Ir leftward, Il rightward
        cost = np.count_nonzero(np.not_equal(Ir_binary_pattern[:, :-i, :, :],Il_binary_pattern[:, i:, :, :]), axis=-1) # shape = (h, w-i, ch)

        # pad cost
        cost_r_2_l = np.pad(cost, ((0, 0), (i, 0), (0, 0)), 'edge') # shape (h, w, ch)
        cost_l_2_r = np.pad(cost, ((0, 0), (0, i), (0, 0)), 'edge') # shape (h, w, ch)

        # sum cost thru all channel
        cost_r_2_l = np.sum(cost_r_2_l, axis=-1)
        cost_l_2_r = np.sum(cost_l_2_r, axis=-1)

        Ir_2_Il_cost.append(cost_r_2_l)
        Il_2_Ir_cost.append(cost_l_2_r)

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    Ir_2_Il_cost = np.array(Ir_2_Il_cost) # shape = (max_disp, h, w)
    Il_2_Ir_cost = np.array(Il_2_Ir_cost) # shape = (max_disp, h, w)
    d = -1
    sigma_color = 20
    sigma_space = 20
    for i, cost in enumerate(Ir_2_Il_cost):
        cost = np.expand_dims(cost, axis=-1).astype(np.float32)
        refined_cost = xip.jointBilateralFilter(Il, cost, d, sigma_color, sigma_space)
        Ir_2_Il_cost[i] = refined_cost

    for i, cost in enumerate(Il_2_Ir_cost):
        cost = np.expand_dims(cost, axis=-1).astype(np.float32)
        refined_cost = xip.jointBilateralFilter(Ir, cost, d, sigma_color, sigma_space)
        Il_2_Ir_cost[i] = refined_cost

    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all

    disparity_L2R = np.argmin(Il_2_Ir_cost, axis=0)
    disparity_R2L = np.argmin(Ir_2_Il_cost, axis=0)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering

    ##### Hole Filling
    hole = np.zeros((h, w))
    for h in range(disparity_R2L.shape[0]):
        for w in range(disparity_R2L.shape[1]):
            d = disparity_L2R[h][w]
            if w+d < disparity_R2L.shape[1]:
                hole[h][w] = (d != disparity_R2L[h][w+d])
            else: hole[h][w] = 1

    coords = np.argwhere(hole)

    for (h, w) in coords:
        f_l = None
        f_r = None
        i = w
        while True:
            i -= 1
            if i < 0: 
                f_l = max_disp
                break
            elif hole[h][i]: continue
            else: 
                f_l = disparity_R2L[h][i]
                break
        i = w
        while True:
            i += 1
            if i >= disparity_R2L.shape[1]:
                f_r = max_disp
                break
            elif hole[h][i]: continue
            else:
                f_r = disparity_R2L[h][i]
                break
        disparity_R2L[h][w] = min(f_l, f_r)

    #### Weighted Median Filter
    r = 25
    disparity_R2L = np.expand_dims(disparity_R2L, axis=-1).astype(np.uint8)
    disparity_R2L = xip.weightedMedianFilter(Il.astype(np.uint8), disparity_R2L, r)

    
    labels = disparity_R2L
    return labels.astype(np.uint8)
    