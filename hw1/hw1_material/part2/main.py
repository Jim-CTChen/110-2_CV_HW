import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter
from matplotlib import cm, pyplot as plt

def show(imgs):
    for i in range(len(imgs)):
        plt.figure(i)
        plt.imshow(imgs[i].astype(np.uint8))
    plt.show()

def cal_cost(img1, img2):
    abs_cost = np.abs(img1.astype(np.int32)-img2.astype(np.int32))
    return np.sum(abs_cost)

def gray_convert(img, weight):
    rgb = img[:,:,0]*weight[2]+img[:,:,1]*weight[1]+img[:,:,2]*weight[0]
    return rgb


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/2_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    # 1.png
    # sigma_s = 2
    # sigma_r = 0.1
    # rgb_weight = [(0, 0, 1), (0, 1, 0), (0.1, 0, 0.9), (0.1, 0.4, 0.5), (0.8, 0.2, 0)]

    # 2.png
    sigma_s = 1
    sigma_r = 0.05
    rgb_weight = [(0.1,0.0,0.9), (0.2,0.0,0.8), (0.2,0.8,0.0), (0.4,0.0,0.6), (1.0,0.0,0.0)]

    JBF = Joint_bilateral_filter(sigma_r=sigma_r, sigma_s=sigma_s)
    jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray)
    bf_out  = JBF.joint_bilateral_filter(img_rgb, img_rgb)
    cost = cal_cost(jbf_out, bf_out)
    cost = np.sum(cost)
    print(f'cost: {cost} for default gray scale convertion')
    for weight in rgb_weight:
        img_gray = gray_convert(img, weight)
        jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray)
        bf_out  = JBF.joint_bilateral_filter(img_rgb, img_rgb)
        cost = cal_cost(jbf_out, bf_out)
        cost = np.sum(cost)
        print(f'cost: {cost} for weight: {weight}')
        show([jbf_out])

    # img_gray = gray_convert(img, rgb_weight[3])
    # jbf_out = JBF.joint_bilateral_filter(img_rgb, img_gray)
    # plt.figure(1)
    # plt.imshow(img_gray, cmap='gray')
    # bf_out  = JBF.joint_bilateral_filter(img_rgb, img_rgb)
    # cost = cal_cost(jbf_out, bf_out)
    # cost = np.sum(cost)
    # print(f'cost: {cost} for weight: {rgb_weight[3]}')
    # show([jbf_out])
        

    # show_multiple([jbf_out, bf_out])

    


if __name__ == '__main__':
    main()