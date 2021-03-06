import calibration
import cv2
import numpy as np
import pickle
import glob
import matplotlib.pyplot as plt
import argparse
import csv

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip


# Global values
# Data read from calibration script
data = pickle.load(open("calibration.p", 'rb'))
mtx = data["mtx"]
dist = data["dist"]

# Data collected after carefully looking at pixels of an undistorted image with straight lanes
src = np.float32([[580, 460], [707, 460], [187, 720], [1125, 720]])
dst = np.float32([[200, 0], [1000, 0], [200, 720], [1000, 720]])

initialized = False
write_counter = 0

class Lane:

    def __init__(self):

        self.detected = False                               # was the line detected in the last iteration?
        self.recent_xfitted = []                            # x values of the last n fits of the line
        self.bestx = None                                   # avg x values of the fitted line over the last n iters
        self.best_fit = None                                # polynomial coeffs averaged over the last n iterations
        self.current_fit = []                               # polynomial coeffs for the most recent fit
        self.radius_of_curvature = None                     # radius of curvature of the line in some units
        self.line_base_pos = None                           # distance in meters of vehicle center from the line
        self.diffs = np.array([0, 0, 0], dtype='float')     # difference in fit coefficients between last and new fits
        self.allx = None                                    # x values for detected line pixels
        self.ally = None                                    # y values for detected line pixels
        self.n = 5                                          # number of iterations to keep memory for
        self.ploty = None                                   # storing the plotting y dimension
        self.ym_per_pix = 30/720                            # meters per pixel in y dimension
        self.xm_per_pix = 3.7/700                           # meters per pixel in x dimension

    def update_poly(self, poly):
        self.current_fit.append(poly)
        if len(self.current_fit) > self.n:
            self.current_fit.pop(0)
        if len(self.current_fit) > 1:
            self.diffs = self.current_fit[-1] - self.current_fit[-2]
        self.best_fit = np.mean(self.current_fit, axis=0)

        # Evaluating curvature
        fit = poly
        y_eval = np.max(self.ploty)
        x = fit[0] * self.ploty ** 2 + fit[1] * self.ploty + fit[2]
        fit_cr = np.polyfit(self.ploty * self.ym_per_pix, x * self.xm_per_pix, 2)
        self.radius_of_curvature = np.sqrt((1 + (2 * fit_cr[0] * y_eval * self.ym_per_pix + fit_cr[1])**2)**3)/np.absolute(2*fit_cr[0])

    def update_x(self, xfit):
        self.recent_xfitted.append(xfit)
        if len(self.recent_xfitted) > self.n:
            self.recent_xfitted.pop(0)
        diff = np.sum(np.square(self.diffs))

        if diff > 500:
            self.recent_xfitted.clear()
            self.recent_xfitted.append(self.bestx)
            self.current_fit.clear()
            self.current_fit.append(self.best_fit)
        else:
            self.bestx = np.mean(self.recent_xfitted, axis=0)
    def update_pix(self, x, y):
        self.allx = x
        self.ally = y

    def get_outputs(self):
        return self.bestx, self.allx, self.ally

LEFT = Lane()
RIGHT = Lane()


def warp(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return warped, Minv


def filter_image(img,
                 s_thresh=(170, 255),
                 sx_thresh=(20, 100),
                 sy_thresh=(20, 100),
                 mag_thresh=(100, 255),
                 dir_thresh=(np.pi/3, np.pi/2)):

    img = np.copy(img)

    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]

    # Threshold color channels
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= 50) & (l_channel <= 255)] = 1

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
    combined[(((sybinary == 1) & (sxbinary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (s_binary == 1)) & (l_binary == 1)] = 1

    return color_binary, combined


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

    # Set parameters for window search
    nwindows = 9  # Choose the number of sliding windows
    margin = 50  # Set the width of the windows +/- margin
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

    # Update lanes
    LEFT.update_pix(leftx, lefty)
    RIGHT.update_pix(rightx, righty)

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):

    global LEFT, RIGHT

    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    LEFT.ploty = ploty
    RIGHT.ploty = ploty

    # Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    LEFT.update_poly(left_fit)
    RIGHT.update_poly(right_fit)

    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    LEFT.update_x(left_fitx)
    RIGHT.update_x(right_fitx)

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return out_img, left_fitx, right_fitx, ploty, leftx, lefty, rightx, righty


def fit_poly(img_shape, leftx, lefty, rightx, righty):

    global LEFT, RIGHT

    # Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    LEFT.update_poly(left_fit)
    RIGHT.update_poly(right_fit)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])

    # Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    LEFT.update_x(left_fitx)
    RIGHT.update_x(right_fitx)

    return left_fitx, right_fitx, ploty


def search_around_poly(binary_warped):

    global LEFT, RIGHT

    # Margin to search around previous polynomial
    margin = 30

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Setting the search area based on previous frame's polynomial
    left_poly = LEFT.best_fit
    polyleft = left_poly[0] * nonzeroy ** 2 + left_poly[1] * nonzeroy + left_poly[2]
    left_lane_inds = ((nonzerox >= polyleft - margin) & (nonzerox < polyleft + margin))
    right_poly = RIGHT.best_fit
    polyright = right_poly[0] * nonzeroy ** 2 + right_poly[1] * nonzeroy + right_poly[2]
    right_lane_inds = ((nonzerox >= polyright - margin) & (nonzerox < polyright + margin))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    # Update lanes
    LEFT.update_pix(leftx, lefty)
    RIGHT.update_pix(rightx, righty)

    op_lfx, op_lx, op_ly = LEFT.get_outputs()
    op_rfx, op_rx, op_ry = RIGHT.get_outputs()

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result, op_lfx, op_rfx, LEFT.ploty, leftx, lefty, rightx, righty


def detect_lanes_standalone(img, kernel=np.ones((3, 3), np.uint8), src=src, dst=dst):

    global LEFT, RIGHT, initialized

    # Undistort incoming image
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Filter and obtain best image for lane detection
    color, comb = filter_image(undist)
    closed_bin = cv2.morphologyEx(src=comb, op=cv2.MORPH_CLOSE, kernel=kernel)
    opened_bin = cv2.morphologyEx(src=closed_bin, op=cv2.MORPH_OPEN, kernel=kernel)

    # Warp the image according to previously decided image points
    warped, Minv = warp(opened_bin, src, dst)
    fitted, lfx, rfx, ply, lx, ly, rx, ry = fit_polynomial(warped)

    # Draw lane markings and lane area
    overlaid = draw_lanes(image=undist, warped=warped, Minv=Minv, left_fitx=lfx, right_fitx=rfx, ploty=ply, leftx=lx,
                          lefty=ly, rightx=rx, righty=ry)

    # Write outputs to files
    cv2.imwrite("./temp/image_overlaid.jpg", overlaid)
    cv2.imwrite("./temp/image_undisted.jpg", undist)
    cv2.imwrite("./temp/image_filtered.jpg", 255*closed_bin)

    return overlaid, comb, closed_bin, opened_bin, warped


def detect_lanes_video(img, kernel=np.ones((3, 3), np.uint8), src=src, dst=dst):

    global LEFT, RIGHT, initialized, write_counter

    # Undistort incoming image
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    # Filter and obtain best image for lane detection
    color, comb = filter_image(undist)
    closed_bin = cv2.morphologyEx(src=comb, op=cv2.MORPH_CLOSE, kernel=kernel)
    opened_bin = cv2.morphologyEx(src=closed_bin, op=cv2.MORPH_OPEN, kernel=kernel)

    # Warp the image according to previously decided image points
    warped, Minv = warp(opened_bin, src, dst)

    if not initialized:
        fitted, lfx, rfx, ply, lx, ly, rx, ry = fit_polynomial(warped)
        initialized = True
    else:
        fitted, lfx, rfx, ply, lx, ly, rx, ry = search_around_poly(warped)

    # Prints
    # 1. Curvature
    curvature = (RIGHT.radius_of_curvature + LEFT.radius_of_curvature)/2
    # 2. Offset of vehicle from center
    offset = (img.shape[1] - (LEFT.recent_xfitted[-1][-1] + RIGHT.recent_xfitted[-1][-1]))/2 * LEFT.xm_per_pix

    # Draw lane markings and lane area
    overlaid = draw_lanes(image=undist, warped=warped, Minv=Minv, left_fitx=lfx, right_fitx=rfx, ploty=ply, leftx=lx,
                          lefty=ly, rightx=rx, righty=ry, offset=offset, curvature=curvature)

    # Write outputs to files
    cv2.imwrite("./processing/overlaid/image_" + str(write_counter) + "_overlaid.jpg",
                cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR))                                      # write final overlay
    """
    cv2.imwrite("./processing/undisted/image_" + str(write_counter) + "_undisted.jpg",
            cv2.cvtColor(undist, cv2.COLOR_RGB2BGR))                                            # write final overlay
    cv2.imwrite("./processing/original/image_" + str(write_counter) + "_original.jpg",
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR))                                           # write original
    cv2.imwrite("./processing/filtered/image_" + str(write_counter) + "_filtered.jpg",
                255*closed_bin)                                                                 # write final overlay
    """
    write_counter += 1
    return overlaid


def draw_lanes(image, warped, left_fitx, right_fitx, ploty, Minv, leftx, lefty, rightx, righty, offset=None, curvature=None):

    # Prepare images to be used for drawing
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    lane_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    bounds_warp = lane_warp

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(lane_warp, np.int_([pts]), (0, 255, 0))
    bounds_warp[lefty, leftx] = [255, 0, 0]
    bounds_warp[righty, rightx] = [0, 0, 255]

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    lane_warped = cv2.warpPerspective(lane_warp, Minv, (image.shape[1], image.shape[0]))
    bounds_warped = cv2.warpPerspective(bounds_warp, Minv, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1.0, lane_warped, 0.3, 0)
    result = cv2.addWeighted(result, 1.0, bounds_warped, 0.3, 0)

    text1 = ""
    if curvature:
        if curvature > 5000:
            text1 = "Straight road ahead"
        else:
            text1 = "Radius of curvature = " + str(curvature) + " meters"
        text2 = "Vehicle is " + str(offset) + " meters from center of lane"
        result = cv2.putText(img=result, text=text1, org=(100, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale=0.75, color=(255, 255, 255), thickness=2)
        result = cv2.putText(img=result, text=text2, org=(100, 130), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale=0.75, color=(255, 255, 255), thickness=2)

    return result


if __name__ == "__main__":

    # Parse from command line
    parser = argparse.ArgumentParser(description="Detect lane markers and drivable space from video. Provide only one of the below two options")
    parser.add_argument('-v', '--video', dest="video", help='Input file to be processed')
    parser.add_argument('-s', '--standalone', dest="standalone", help='Run pipeline on a single example image')
    args = parser.parse_args()

    if not args.video and not args.standalone:
        print("Didn't provide an input video or a standalone image.")
        exit()

    if args.video and args.standalone:
        print("Provided both options. Rerun and choose one.")
        exit()

    if args.video:
        vid = args.video
        video_name = vid[:vid.rfind(".")]

        # Read Video
        output = 'output_' + video_name + '.mp4'
        clip1 = VideoFileClip(vid)
        white_clip = clip1.fl_image(detect_lanes_video)  # NOTE: this function expects color images!!
        white_clip.write_videofile(output, audio=False)

    if args.standalone:#
        img_path = args.standalone
        img_name = img_path[img_path.rfind("/")+1:]
        img = cv2.imread(img_path)
        cv2.imshow(img_name, img)
        #cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, comb, opened, closed, warped = detect_lanes_standalone(img)
        """
        cv2.imwrite("./output_images/filtered_" + img_name, 255*comb)
        cv2.imwrite("./output_images/opened_" + img_name, 255*opened)
        cv2.imwrite("./output_images/closed_" + img_name, 255*closed)
        cv2.imwrite("./output_images/warped_" + img_name, 255*warped)
        cv2.imwrite("./output_images/output_" + img_name, img)

        cv2.imshow("./output_images/output_" + img_name, img)
        cv2.imshow("./output_images/filtered_" + img_name, 255*comb)
        cv2.imshow("./output_images/opened_" + img_name, 255*opened)
        cv2.imshow("./output_images/closed_" + img_name, 255*closed)
        cv2.imshow("./output_images/warped_" + img_name, 255*warped)
        cv2.waitKey(0)
        """
