from cProfile import label
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt

def show(img):
    plt.figure(0)
    plt.imshow(img.astype(np.uint8))
    plt.show()

def show_multiple(imgs):
    for i in range(len(imgs)):
        plt.figure(i)
        plt.imshow(imgs[i].astype(np.uint8))
    plt.show()

class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
        self.kernel_r = 3*sigma_s # kernal half window size, kernel = (-r, r) x (-r, r)
        self.s_kernel = np.stack(np.array([ 
            [math.exp(-((i**2+j**2)/(2*self.sigma_s**2))) for j in range(-self.kernel_r, self.kernel_r+1)] 
            for i in range(-self.kernel_r, self.kernel_r+1)
        ]))
        # LUT for kernel
        # range from 0.0~255.0
        self.LUT = np.array([math.exp(-(diff/255)**2/(2*self.sigma_r**2)) for diff in range(0, 256)])

    def weight_LUT(self, diff):
        mapping = lambda x : self.LUT[(abs(x))]
        return mapping(diff)

    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        # print(f'img shape: {img.shape}')
        # print(guidance.shape)
        # print(f'padded img shape: {padded_img.shape}')
        # print(padded_guidance.shape)

        ### TODO ###
        JBF = 0;
        BF = 1;
        type = BF if padded_guidance.ndim != 2 else JBF
        
        output = np.zeros((img.shape[0], img.shape[1], 3))
        weight = np.zeros((img.shape[0], img.shape[1], 3))

        r = self.kernel_r
        w = img.shape[1]
        h = img.shape[0]

        if type == JBF:
            for i in range(-r, r+1):
                for j in range(-r, r+1):
                    # range kernel (guidance)
                    diff = padded_guidance[r+i:r+i+h,r+j:r+j+w]-padded_guidance[r:r+h, r:r+w] # shape = (h, w)
                    r_kernel = self.weight_LUT(diff)
                    kernel = r_kernel*self.s_kernel[r+i][r+j]       # shape = (h, w)
                    kernel = np.stack((kernel,)*3, axis=-1)         # shape = (h, w, 3)
                    out = kernel*padded_img[r+i:r+i+h,r+j:r+j+w]    # shape = (h, w, 3)
                    weight += kernel # shape = (h, w, 3)
                    output += out
        else:
            for i in range(-r, r+1):
                for j in range(-r, r+1):
                    # range kernel (guidance)
                    diff = padded_guidance[r+i:h+r+i,r+j:w+r+j]-padded_guidance[r:r+h, r:r+w] # shape = (h, w, 3)
                    r_kernel = self.weight_LUT(diff)
                    r_kernel = np.prod(r_kernel, axis=-1)           # shape = (h, w), prod three channels
                    kernel = r_kernel*self.s_kernel[i+r][j+r]       # shape = (h, w)
                    kernel = np.stack((kernel,)*3, axis=-1)         # shape = (h, w, 3)
                    out = kernel*padded_img[r+i:r+i+h,r+j:r+j+w]    # shape = (h, w, 3)
                    weight += kernel # shape = (h, w, 3)
                    output += out
        
        output /= weight # scale back
        
        # show_multiple([img, output.astype(np.uint8)])
        
        return np.clip(output, 0, 255).astype(np.uint8)