import calibration
import cv2
import numpy as np
import pickle
import glob


class Lane:
    def __init__(self):
        self.detected = False                           # was the line detected in the last iteration?
        self.recent_xfitted = []                        # x values of the last n fits of the line
        self.bestx = None                               # average x values of the fitted line over the last n iterations
        self.best_fit = None                            # polynomial coefficients averaged over the last n iterations
        self.current_fit = [np.array([False])]          # polynomial coefficients for the most recent fit
        self.radius_of_curvature = None                 # radius of curvature of the line in some units
        self.line_base_pos = None                       # distance in meters of vehicle center from the line
        self.diffs = np.array([0,0,0], dtype='float')   # difference in fit coefficients between last and new fits
        self.allx = None                                # x values for detected line pixels
        self.ally = None                                # y values for detected line pixels


def abs_sobel_thresh(gray, orient='x', ksize=3, thresh=(0, 255)):
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    grad_binary = np.zeros_like(gray)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary


def mag_thresh(gray, ksize=3, mag_thresh=(0, 255)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    scaled_sobel = np.uint8(255 * mag_sobel / np.max(mag_sobel))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return mag_binary


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sobel_dir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(gray)
    dir_binary[(sobel_dir >= thresh[0]) & (sobel_dir <= thresh[1])] = 1

    return dir_binary


def hls(img, thresh=(200,255), ksize=3):
    # Saturation channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    sat = hls[:, :, 2]
    sat_binary = np.zeros_like(sat)
    sat_binary[(sat > thresh[0]) & (sat <= thresh[1])] = 1

    return sat_binary


def warp(img, bb):
    return img
    """
    src = np.float32([corners[0], corners[7], corners[40], corners[47]])
    dst = np.float32([[100,100],[1200,100],[100,860],[1200,860]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(undist, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    """


def find_lanes(img):
    return None, None


if __name__=="__main__":
    data = pickle.load(open("calibration.p", 'rb'))
    mtx = data["mtx"]
    dist = data["dist"]
    for filename in glob.glob("./test_images/*jpg"):
        img = cv2.imread(filename)
        undist = cv2.undistort(img, mtx, dist, None, mtx)
        # cv2.imshow('Before', img)
        # cv2.imshow('After', undist)
        # cv2.waitKey(0)

        # Filters
        ksize = 3
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        sat_binary = hls(img)
        gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(0, 255))
        grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(0, 255))
        mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(0, 255))
        dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0, np.pi / 2))

        # Warp binary image with mask extremities to make it orthographic
        warped_img = warp(S, None)

        # Find pixels of interest on L&R and fit a polynomial
        L, R = find_lanes(warped_img)

        # Unwarp the polynomial in camera image
        # Add markers as needed
        # Distort image back to original camera
        # For video, find_lanes changes

        # Plot away
        cv2.imshow('Saturation', 255*S)
        # cv2.imshow('After', undist)
        cv2.waitKey(0)