---
layout: post
title: Lane Detection Basic 
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

This project demonstrate how to detect lane lines in images using Python and OpenCV. OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

Below is an example for land detection:

<img src="https://raw.githubusercontent.com/jiajuns/LaneDetectionBasic/master/examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

To start
---
**Step 1:** Set up the [CarND Term1 Starter Kit](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/83ec35ee-1e02-48a5-bdb7-d244bd47c2dc/lessons/8c82408b-a217-4d09-b81d-1bda4c6380ef/concepts/4f1870e0-3849-43e4-b670-12e6f2d4b7a7) if you haven't already.

**Step 2:** Open the code in a Jupyter Notebook [P1.ipynb](https://github.com/jiajuns/LaneDetectionBasic/blob/master/P1.ipynb)


Method Description
---
The pipeline consists of 5 steps.

1. Convert images to grayscale,then apply gaussian blur to smooth the image.

    ```python
    def grayscale(img):
        """Applies the Grayscale transform
        This will return an image with only one color channel
        but NOTE: to see the returned image as grayscale
        (assuming your grayscaled image is called 'gray')
        you should call plt.imshow(gray, cmap='gray')"""
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Or use BGR2GRAY if you read an image with cv2.imread()
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(img, kernel_size):
        """Applies a Gaussian Noise kernel"""
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    kernel_size = 5
    gray = grayscale(image)
    blur_gray = gaussian_blur(gray, kernel_size)
    ```

2. Obtain edges using canny edge detection.

    ```python
    def canny(img, low_threshold, high_threshold):
        """Applies the Canny transform"""
        return cv2.Canny(img, low_threshold, high_threshold)

    low_threshold = 80
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)
    ```

3. Apply a mask to clean up the edges that are not in the interests region.

    ```python
    def region_of_interest(img, vertices):
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image



    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   

    # define a four sided polygon to mask
    imshape = img.shape
    vertices = np.array([[(0,imshape[0]), (imshape[1]/2, 3*imshape[0]/5), (imshape[1]/2, 3*imshape[0]/5), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    ```

4. Use hough transformation to fit lanes to the image.

    ```python
    def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
        """
        `img` should be the output of a Canny transform.
            
        Returns an image with hough lines drawn.
        """
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        draw_lines(line_img, lines)
        
        return line_img

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 30     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 30 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    line_img = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)
    line_img = region_of_interest(line_img, vertices)
    ```

5. Draw fitted lanes onto the raw image.
```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    imshape = img.shape
    for line in lines:
        for x1,y1,x2,y2 in line:
            k = (y2-y1)/(x2-x1)
            b = y2 - k*x2
            try:
                extra_x1 = int((imshape[0] - b)/k)
                extra_x2 = int((-b)/k)
                cv2.line(img, (extra_x1, imshape[0]), (extra_x2, 0), color, thickness)
            except:
                continue
```
