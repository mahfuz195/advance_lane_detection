## Advance Lane Detection

### The goal of the project is to write a software pipeline to identify the lane boundaries in a video, measure the radius of the road, and measure the offset of the car position from the center position of a lane.
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

[image1]: ./CarND-Advanced-Lane-Lines/camera_calibration.png "Undistorted"
[image2]: ./CarND-Advanced-Lane-Lines/examples/original_image.png "Road Transformed"
[image3]: ./CarND-Advanced-Lane-Lines/examples/undistorted_image.png "Road Transformed"
[image4]: ./CarND-Advanced-Lane-Lines/examples/binary_image.png "Binary Example"
[image5]: ./CarND-Advanced-Lane-Lines/examples/warped_image.png "Warped Image"
[image6]: ./CarND-Advanced-Lane-Lines/examples/histogram.png "Histogram Image"
[image7]: ./CarND-Advanced-Lane-Lines/examples/poly.png "Ploy Image"
[image8]: ./CarND-Advanced-Lane-Lines/examples/anotated.png "Anotated Image"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code cell (1-4) of the IPython notebook located in "pipeline.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
![alt text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I used the following lines of code to perform a prespective transform using following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 220, 720      | 320, 720      | 
| 1200, 720     | 920, 720      |
| 570, 470      | 320, 1        |
| 780, 470      | 920, 1        |

```
dst = np.float32([bottom_left,bottom_right,top_right,top_left])
M = cv2.getPerspectiveTransform(source, dst)
M_inv = cv2.getPerspectiveTransform(dst, source)
img_size = (image_shape[1], image_shape[0])
warped = cv2.warpPerspective(thresholded, M, img_size , flags=cv2.INTER_LINEAR)
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
I did a histogram analysis of the binary image and took the maximum picks on the right side and left side to detect the staring points of the left and right lanes respectively. 

![alt text][image6]

Then I did a sliding window search with a window size of 9 and fit my lane lines with a 2nd order polynomial. Which produced the following output:

![alt text][image7]



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Using `measure_radius_of_curvature` I implemented the calculation of measuring the radis. Here is the funciton that I have used:
```
def measure_radius_of_curvature(x_values):
    ym_per_pix = 30/720 
    xm_per_pix = 3.7/700 
    y_points = np.linspace(0, num_rows-1, num_rows)
    y_eval = np.max(y_points)
    fit_cr = np.polyfit(y_points*ym_per_pix, x_values*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented putting all the pieces together and produced the anotated image as below:

![alt text][image8]



---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
