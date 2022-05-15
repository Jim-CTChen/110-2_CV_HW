import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import coord_to_homogeneous_coord, homogeneous_coord_to_coord, solve_homography, transform_with_H, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None
    ransac_it = 5000
    ransac_distance_threshold = 12

    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        img1 = imgs[idx]
        img2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1, des2)
        # matches = sorted(matches, key = lambda x:x.distance) # sort them in the order of their distance

        # TODO: 2. apply RANSAC to choose best H
        max_inlier = 0
        best_H = None
        img1_kp = np.array([kp1[m.queryIdx].pt for m in matches]).astype(int)
        img1_kp_h = coord_to_homogeneous_coord(img1_kp)
        img2_kp = np.array([kp2[m.trainIdx].pt for m in matches]).astype(int)
        img2_kp_h = coord_to_homogeneous_coord(img2_kp)
        for i in range(ransac_it):
            selected_matches = random.sample(matches, 4) # random select four matches to compute H

            u = np.array([kp2[m.trainIdx].pt for m in selected_matches]).astype(int)
            v = np.array([kp1[m.queryIdx].pt for m in selected_matches]).astype(int)
            H = solve_homography(u, v)
            
            projected_pt = homogeneous_coord_to_coord(transform_with_H(H, img2_kp_h))
            error = np.sqrt(np.sum(np.square(img1_kp-projected_pt), axis=1))
            if np.count_nonzero(error < ransac_distance_threshold) > max_inlier:
                max_inlier = np.count_nonzero(error < ransac_distance_threshold)
                best_H = H

        # TODO: 3. chain the homographies
        last_best_H = last_best_H @ best_H

        # TODO: 4. apply warping
        dst = warping(img2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')

    out = dst
    return out

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)