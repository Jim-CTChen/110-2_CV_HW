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
        s_kernel = np.stack(np.array([ 
            [math.exp(-((i**2+j**2)/(2*self.sigma_s**2))) for j in range(-self.kernel_r, self.kernel_r+1)] 
            for i in range(-self.kernel_r, self.kernel_r+1)
        ]))
        self.s_kernel = np.stack((s_kernel, )*3, axis=-1)
        # LUT for kernel
        # range from 0.0~255.0
        self.LUT = np.array([math.exp(-(diff/2550)**2/(2*self.sigma_r**2)) for diff in range(0, 255*10+1)])
    
    def LUT_for_kernel_JB(self, diff):
        '''
            joint bilateral (gray scale)
            convert f'(x-i, y-j) - f'(x, y) to LUT
            diff.shape = (3,)
        '''
        abs_diff = np.abs(diff)*10
        kernel_weight = np.array((self.LUT[abs_diff[0]],)*3)
        return kernel_weight

    def LUT_for_kernel_B(self, diff):
        '''
            bilateral (rgb)
            convert f'(x-i, y-j) - f'(x, y) to LUT
            diff.shape = (3,)
        '''
        abs_diff = np.abs(diff)*10
        weight = self.LUT[abs_diff[0]]*self.LUT[abs_diff[1]]*self.LUT[abs_diff[2]]
        kernel_weight = np.array((weight,)*3)
        # print(kernel_weight)
        return kernel_weight


    def joint_bilateral_filter(self, img, guidance):
        print(f'img shape: {img.shape}')
        # print(guidance.shape)
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        print(f'padded img shape: {padded_img.shape}')
        # print(padded_guidance.shape)

        ### TODO ###

        # for gray scale guidance
        # duplicate for 3 channels
        JBF = 0;
        BF = 1;
        type = BF
        if padded_guidance.ndim == 2:
            type = JBF
            padded_guidance = np.stack((padded_guidance,)*3, axis=-1)
        
        output = np.zeros((img.shape[0], img.shape[1], 3))
        r = self.kernel_r
        
        for i in range(self.pad_w, padded_img.shape[0]-self.pad_w):
            for j in range(self.pad_w, padded_img.shape[1]-self.pad_w):
                # print(f'coor: ({i-self.pad_w}, {j-self.pad_w})')
                # range kernel
                r_kernel = 0
                if type == JBF:
                    r_kernel = np.array([
                        [self.LUT_for_kernel_JB(padded_guidance[i+k][j+l]-padded_guidance[i][j]) for l in range(-r, r+1)] 
                        for k in range(-r, r+1)
                    ])
                else:
                    r_kernel = np.array([
                        [self.LUT_for_kernel_B(padded_guidance[i+k][j+l]-padded_guidance[i][j]) for l in range(-r, r+1)] 
                        for k in range(-r, r+1)
                    ])
                # print(r_kernel.shape)
                kernel = self.s_kernel*r_kernel # shape (self.wndw, self.wndw, 3)
                kernel_weight = np.sum(np.sum(kernel, axis=0), axis=0)
                kernel_result = kernel*padded_img[i-r:i+r+1, j-r:j+r+1]
                output[i-self.pad_w][j-self.pad_w] = np.sum(np.sum(kernel_result, axis=0), axis=0)/kernel_weight
        
        show_multiple([img, output.astype(np.uint8)])
        
        return np.clip(output, 0, 255).astype(np.uint8)