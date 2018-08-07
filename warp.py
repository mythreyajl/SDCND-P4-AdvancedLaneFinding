import cv2
import numpy as np


def warp(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return warped, Minv


src = np.float32([[580, 460], [707, 460], [187, 720], [1125, 720]])
dst = np.float32([[200, 0], [1000, 0], [200, 720], [1000, 720]])
image = cv2.imread("./test_images/straight_lines1.jpg")
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
warped = warp(img=image2, src=src, dst=dst)

cv2.imshow('original', image)
cv2.imshow('warped', 255*warped[0])

cv2.imwrite('./output_images/warped_straight_lines1.jpg', 255*warped[0])
cv2.waitKey(0)