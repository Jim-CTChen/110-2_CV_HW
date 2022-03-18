import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian
from matplotlib import pyplot as plt


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=5.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/2.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32) # gray scale

    # print origin img
    origin_img = cv2.imread(args.image_path)
    # plt.figure(0)
    # plt.imshow(cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB))
    # plt.show()
    

    ### TODO ###
    print('current threshold: 2')
    DoG = Difference_of_Gaussian(2)
    keypoints = DoG.get_keypoints(img)
    print(f'total keypoints: {keypoints.shape[0]}')
    plot_keypoints(img, keypoints, './2_png_threshold_2.jpg')

    print('current threshold: 5')
    DoG = Difference_of_Gaussian(5)
    keypoints = DoG.get_keypoints(img)
    print(f'total keypoints: {keypoints.shape[0]}')
    plot_keypoints(img, keypoints, './2_png_threshold_5.jpg')

    print('current threshold: 7')
    DoG = Difference_of_Gaussian(7)
    keypoints = DoG.get_keypoints(img)
    print(f'total keypoints: {keypoints.shape[0]}')
    plot_keypoints(img, keypoints, './2_png_threshold_7.jpg')


    # draw circle on keypoints
    # radius = 5
    # color = (0, 0, 255)
    # thickness = 2
    # img_with_keypoints = origin_img
    # for x, y in keypoints:
    #     img_with_keypoints = cv2.circle(img_with_keypoints, (y, x), radius, color, thickness)
    # plt.figure(1)
    # plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    # plt.show()

if __name__ == '__main__':
    main()