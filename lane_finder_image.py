import calibration
import cv2
import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt

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


def pipeline(img,
             s_thresh=(170, 255),
             sx_thresh=(20, 100),
             sy_thresh=(20, 100),
             mag_thresh=(100, 255),
             dir_thresh=(np.pi/3, np.pi/2)):

    img = np.copy(img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobelx)
    sxbinary[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

    # Sobel y
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1)  # Take the derivative in x
    abs_sobely = np.absolute(sobely)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))
    sybinary = np.zeros_like(scaled_sobely)
    sybinary[(scaled_sobely >= sy_thresh[0]) & (scaled_sobely <= sy_thresh[1])] = 1

    # Magnitude of gradient
    mag_sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    mag_scaled = np.uint8(255 * mag_sobel / np.max(mag_sobel))
    mag_binary = np.zeros_like(mag_scaled)
    mag_binary[(mag_scaled >= mag_thresh[0]) & (mag_scaled <= mag_thresh[1])] = 1

    # Direction of gradient
    dir_sobel = np.absolute(np.arctan2(abs_sobely, abs_sobelx))
    dir_scaled = np.uint8(255 * dir_sobel / np.max(dir_sobel))
    dir_binary = np.zeros_like(dir_scaled)
    dir_binary[(dir_sobel >= dir_thresh[0]) & (dir_sobel <= dir_thresh[1])] = 1

    # Stack each channel
    color_binary = np.dstack((sybinary, sxbinary, s_binary)) * 255
    combined = np.zeros_like(sybinary)
    #combined[((sybinary == 1) & (sxbinary == 1)) | (s_binary == 1)] = 1
    combined[((sybinary == 1) & (sxbinary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)] = 1
    return color_binary, combined


def warp(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return warped, Minv


def find_lane_pixels(binary_warped):

    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    nwindows = 9  # Choose the number of sliding windows
    margin = 100  # Set the width of the windows +/- margin
    minpix = 50   # Set minimum number of pixels found to recenter window

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        # Current bottom boundaries  of the window for the two lanes
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identification of the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # Recenter left and right lanes based on new information
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fitx, right_fitx, ploty


def draw_lanes(image, warped, left_fitx, right_fitx, ploty, Minv):
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return result


def find_lanes(img):
    return None, None


if __name__ == "__main__":
    data = pickle.load(open("calibration.p", 'rb'))
    mtx = data["mtx"]
    dist = data["dist"]
    for filename in glob.glob("./undist_test_images/*jpg"):
        # Read and apply undistortion
        undist = cv2.imread(filename)
        # undist = cv2.undistort(img, mtx, dist, None, mtx)

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
        # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (sat_binary == 1)] = 1
        kernel = np.ones((3, 3), np.uint8)
        opened_bin = cv2.morphologyEx(src=combined, op=cv2.MORPH_OPEN, kernel=kernel)
        closed_bin = cv2.morphologyEx(src=combined, op=cv2.MORPH_CLOSE, kernel=kernel)

        color, comb = pipeline(undist)
        kernel = np.ones((3, 3), np.uint8)
        closed_bin = cv2.morphologyEx(src=comb, op=cv2.MORPH_CLOSE, kernel=kernel)
        opened_bin = cv2.morphologyEx(src=closed_bin, op=cv2.MORPH_OPEN, kernel=kernel)

        # Warp binary image with mask extremities to make it orthographic
        src = np.float32([[540, 490], [750, 490], [187, 720], [1125, 720]])
        dst = np.float32([[200, 0], [1000, 0], [200, 720], [1000, 720]])
        warped_img_a, _ = warp(closed_bin, src, dst)
        warped_img_b, Minv = warp(opened_bin, src, dst)
        warped_img_c, _= warp(comb, src, dst)

        # Find pixels of interest on L&R and fit a polynomial
        L, R = find_lanes(warped_img_c)

        # Perspective change the polynomial in camera image
        # Add markers as needed
        # Distort image back to original camera
        # For video, find_lanes changes

        # Plot away
        img_name = filename[filename.rfind('/')+1:]
        # cv2.imshow('Closed: ' + img_name, 255*closed_bin)
        # cv2.imshow('Opened: ' + img_name, 255*opened_bin)
        # cv2.imshow('Satted: ' + img_name, 255*comb)
        # cv2.imshow(img_name, 255 * warped_img_c)
        # cv2.imshow('Closed: ' + img_name, 255*warped_img_a)
        # cv2.imshow('Saturation: ' + img_name, 255*warped_img_c)
        fitted, lfx, rfx, ply = fit_polynomial(warped_img_b)
        out = draw_lanes(image=undist, warped=warped_img_b, Minv=Minv, left_fitx=lfx, right_fitx=rfx, ploty=ply)

        #cv2.imshow('Unclosed: ' + img_name, 255 * warped_img_b)
        #cv2.imshow('Fit: ' + img_name,  out)

        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        plt.imshow(out)

        plt.show()
        plt.close()
        #cv2.waitKey(0)
