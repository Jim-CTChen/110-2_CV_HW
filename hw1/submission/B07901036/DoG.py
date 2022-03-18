from concurrent.futures import thread
from mimetypes import guess_all_extensions
from cv2 import threshold
import numpy as np
import cv2
from matplotlib import pyplot as plt

def show(img):
    plt.figure(0)
    plt.imshow(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.show()

def show_multiple(imgs):
    for i in range(len(imgs)):
        plt.figure(i)
        plt.imshow(cv2.cvtColor(imgs[i].astype(np.uint8), cv2.COLOR_BGR2RGB))
    plt.show()

def check_extreme(kernel):
    max_flag = True
    min_flag = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if max_flag:
                    max_flag = kernel[1][1][1] >= kernel[i][j][k]
                if min_flag:
                    min_flag = kernel[1][1][1] <= kernel[i][j][k]
    return max_flag or min_flag


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        # first octaves
        gaussian_images_o1 = [image]
        
        for i in range(self.num_DoG_images_per_octave):
            gaussian_images_o1.append(cv2.GaussianBlur(gaussian_images_o1[0], ksize=(0, 0), sigmaX=self.sigma**(i+1), sigmaY=self.sigma**(i+1)))

        # display result of Guassian Blur octave 1
        # show_multiple(gaussian_images_o1)

        # second octave
        # downsample
        last_img = gaussian_images_o1[-1]
        gaussian_images_o2 = [cv2.resize(last_img, (last_img.shape[1]//2, last_img.shape[0]//2), interpolation=cv2.INTER_NEAREST)]

        for i in range(self.num_DoG_images_per_octave):
            gaussian_images_o2.append(cv2.GaussianBlur(gaussian_images_o2[0], ksize=(0, 0), sigmaX=self.sigma**(i+1), sigmaY=self.sigma**(i+1)))
        # show_multiple(gaussian_images_o2)

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images_o1 = np.array([np.subtract(gaussian_images_o1[i], gaussian_images_o1[i+1]) for i in range(len(gaussian_images_o1)-1)])
        dog_images_o2 = np.array([np.subtract(gaussian_images_o2[i], gaussian_images_o2[i+1]) for i in range(len(gaussian_images_o2)-1)])
        # for report
        # normalize from -255~255 to 0~255
        # dog_images_o1_norm = (dog_images_o1+255)*(256/511)
        # dog_images_o2_norm = (dog_images_o2+255)*(256/511)
        # show_multiple(dog_images_o1_norm)
        # show_multiple(dog_images_o2_norm)


        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = []

        # octave 1
        for i in range(1, dog_images_o1.shape[0]-1):
            for j in range(1, dog_images_o1.shape[1]-1):
                for k in range(1, dog_images_o1.shape[2]-1):
                    if abs(dog_images_o1[i][j][k]) <= self.threshold: continue
                    kernel = dog_images_o1[i-1:i+2, j-1:j+2, k-1:k+2]
                    if check_extreme(kernel):
                        keypoints.append([j, k])

        # octave 2
        for i in range(1, dog_images_o2.shape[0]-1):
            for j in range(1, dog_images_o2.shape[1]-1):
                for k in range(1, dog_images_o2.shape[2]-1):
                    if abs(dog_images_o2[i][j][k]) <= self.threshold: continue
                    kernel = dog_images_o2[i-1:i+2, j-1:j+2, k-1:k+2]
                    if check_extreme(kernel):
                        keypoints.append([j*2, k*2]) # upsample back
        
        keypoints = np.array(keypoints)
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)


        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 

        
        

        return keypoints
