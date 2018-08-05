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


def dir_threshold(gray, ksize=3, thresh=(0, np.pi/2)):
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    sobel_dir = np.arctan2(abs_sobely, abs_sobelx)
    dir_binary = np.zeros_like(gray)
    dir_binary[(sobel_dir >= thresh[0]) & (sobel_dir <= thresh[1])] = 1

    return dir_binary


def hls(img, thresh=(200, 255)):
    # Saturation channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    sat = hls[:, :, 2]
    sat_binary = np.zeros_like(sat)
    sat_binary[(sat > thresh[0]) & (sat <= thresh[1])] = 1

    return sat_binary


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary


def warp(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped


def find_lanes(img):
    return None, None


if __name__=="__main__":
    data = pickle.load(open("calibration.p", 'rb'))
    mtx = data["mtx"]
    dist = data["dist"]
    for filename in glob.glob("./undist_test_images/*jpg"):
        # Read and apply undistortion
        undist = cv2.imread(filename)
        #undist = cv2.undistort(img, mtx, dist, None, mtx)

        # Filters
        k = 3
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        sat_binary = hls(undist, thresh=(180, 255))
        gradx = abs_sobel_thresh(gray, orient='x', ksize=k, thresh=(100, 255))
        grady = abs_sobel_thresh(gray, orient='y', ksize=k, thresh=(100, 255))
        mag_binary = mag_thresh(gray, ksize=k, mag_thresh=(100, 255))
        dir_binary = dir_threshold(gray, ksize=k, thresh=(-np.pi / 2, np.pi / 2))

        # Add Morphological operations to the combined image
        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (sat_binary == 1)] = 1
        kernel = np.ones((3, 3), np.uint8)
        opened_bin = cv2.morphologyEx(src=combined, op=cv2.MORPH_OPEN, kernel=kernel)
        closed_bin = cv2.morphologyEx(src=combined, op=cv2.MORPH_CLOSE, kernel=kernel)

        # Warp binary image with mask extremities to make it orthographic
        src = np.float32([[610, 440], [675, 440], [205, 720], [1120, 720]])
        dst = np.float32([[300, 0], [1000, 0], [300, 720], [1000, 720]])
        warped_img_a = warp(closed_bin, src, dst)
        warped_img_b = warp(combined, src, dst)
        warped_img_c = warp(sat_binary, src, dst)

        # Find pixels of interest on L&R and fit a polynomial
        L, R = find_lanes(warped_img_c)

        # Perspective change the polynomial in camera image
        # Add markers as needed
        # Distort image back to original camera
        # For video, find_lanes changes

        # Plot away
        img_name = filename[filename.rfind('/')+1:]
        # cv2.imshow(img_name, 255 * warped_img_c)
        cv2.imshow('Closed:     ' + img_name, 255*warped_img_a)
        cv2.imshow('Unclosed:   ' + img_name, 255*warped_img_b)
        cv2.imshow('Saturation: ' + img_name, 255*warped_img_c)
        # cv2.imshow('After', undist)
        cv2.waitKey(0)
