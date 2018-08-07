## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/calibration5.jpg "Original"
[image2]: ./camera_cal/undistorted_calibration5.jpg "Undistorted"
[image3]: ./test_images/test1.jpg "Original"
[image4]: ./undist_test_images/test1.jpg "Undistorted"
[image5]: ./output_images/filtered_test1.jpg "Filtered"
[image6]: ./output_images/closed_test1.jpg "Closed"
[image7]: ./output_images/opened_test1.jpg "Opened"
[image8]: ./test_images/straight_lines1.jpg "Original"
[image9]: ./output_images/warped_straight_lines1.jpg "Warped"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Provided this file - writeup.md.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is present in the 'calibration.py' script. The script reads all the images using the glob package, calculates the distortion parameters and the camera matrix and stores them into a pickle file. The way I intended to apply calibration in my project is as an offline, pre-processing step to reduce runtime complexity of the actual algorithm. During runtime, I load the pickle file and just extract the camera matrix and distortion parameters.

The calibration class reads in one image at a time. From this it extracts the corner locations of each of the chessboard corners. These corners are assumed to be in a fixed (x, y, z(=0)) location in the world. The top left interior corner is assumed to be the origin of the world. The width of each of the checkers is assumed to be unit 1. Therefore there is a mapping from each corner in the image to their original locations in the world. A matrix called `imgpoints` stores each of the image locations and another called `objpoints` stores the locations in real world. This is a repeated matrix of `objp`, the original locations of all corners.

These are fed to the `cv2.calibrateCamera()` to solve for distortion parameters, the camera projection matrix. During runtime, I use these parameters as arguments to the `cv2.undistort()` function as a starting step in my pipeline. Here's an example of distortion corrected calibration image:

![alt text][image1]
![alt text][image2]

Also 

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
Here is the undistortion process applied to examples images provided to us: 
![alt text][image3]
![alt text][image4]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I filtered the image based on its x-gradient, y-gradient, gradient-magnitude, gradient-diretion and saturation channel. Following this, I applied morphological operations to 'close' and then 'open' the thus obtained image. The morphological operations helped in removing black holes in lane markings and the white specks elsewhere. I used the closed image in the rest of the algorithm. These are all captured in the function filter_image between line 95 and 144 and in the detect_lanes in lines 366-368.

![alt text][image3]
![alt text][image5]
![alt text][image6]
![alt text][image7]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
I perform warping in a method called `warp()` in lines 87 to 92. In order to warp points, I figured a mapping between a straight image and where it should appear in the perspective transformed image. 
The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 200, 0        | 
| 187, 720      | 200, 720      |
| 1125,720      | 1000, 720     |
| 707, 460      | 1000, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image8]
![alt text][image9]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
