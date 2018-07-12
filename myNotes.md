# Miscellaneous
1. Study glob API for reading images
1. Detect lane lines using masking and thresholding techniques.
1. Perform a perspective transform to get a birds eye view of the lanes.
1. Then we can fit a polynomial to lane line.
1. Extract curvature of the lane using some math.
1. HLS: threshold for s channel is [90, 255] inclusive of upper limit and exclusive of lower limit.



# Color spaces
## HSL, HSV and HSI 
H channel: Hue (color), values range (png) 0-359 and jpg (0-179). Common to all the representations. 
S channel: Saturation, range 0-1 (png) and 0-255 (jpg)
L/V/I channel: Lightness, range 0-1 and 0-255 (jpg)

The definition of S and L/V/I are different in various color spaces. 

### HSV color space
The HSV representation models the way paints of different colors mix together, with the saturation dimension resembling various shades of brightly colored paint, and the value dimension resembling the mixture of those paints with varying amounts of black or white paint.

### HSL color space
The HSL model attempts to resemble more perceptual color models such as NCS or Munsell, placing *fully saturated colors around a circle at a lightness value of ​1⁄2*, where a lightness value of 0 or 1 is fully black or white, respectively.



tunable parameters
1. ksize for sobel filter
1. thresholds for vairous channels
1. inclusion of various channels
1. Assuming 20fps video and 60 miles /hr speed, we can expect to see approximately 30 meter road in 20 frames
hence using a averaging window of length 5-10 frames for all parameters in Lines() class.



---
## Preparation
1. Caliberate the camera to get (mtx, dist)
2. Compute the perspective (and inverse) matrix M (and Minv).

## Pipeline for image processing:
1. distortion correction of the image using (mtx, dist) 
2. Apply combined thresholds to get pixel (or edge) image, i.e., combined_binary.
3. Apply perspective transform M to a combined_binary.
4. Detect the lane lines using 
	- Find lines algorithm-1: 
		- maximum of histogram as lane lines x position.
		- proceed window by window. Collect the pixels within a window. If enough pixels are found in a window, compute mean x-value of the pixels within window and use it as a center x-position for next window. If not enough pixels in current window then use current x-position as center for next window as well.
		- Fit 2nd degree polynomial
	- Find lines algorithm-2: [seems bit more robust]
		- convolve histogram with all ones to average neighboring values.
		- Choose argmax as center for current layer
		- proceed window by window. For each window perform histogram + convolution and choose max in margin around previous center. 
		- Collect the pixels within each window.
		- Fit 2nd degree polynomial  
	- Update lane lines:
		- compute a ROI as wide path with margin around previous estimates of polynomial fit.
		- Find new pixel values in this ROI
		- Fit a new polynomial.
	- Measure curvature and offset from lane center
	- Update the Lines() class for each lane line for each frame with all parameters of importance.
	- 
	- 
5.  - Perform sanity check
		- similar curvature: convert to meters
		- almost parallel
		- sensible distance as a lane width
	- Robustification ideas
		- [sam] merge the lane points with centered offset to get one estimate
		- [sam] estimate lines separately and believe the one that has correct curvature as running average stored in Lines()
		- Reset if Sanity check fails.
			- retain previous frames estimate
			- if failed for multiple frames, perform sliding window hist again.
		- smoothing over multiple frames of estimate to get much smoother outcome.
6. Draw the filled green lane as a drive-able zone.
		- 	


## Todo
1. Implement a basic version of project to check if we qualify the rubrics and then optimize for further improvements.
1. implement lane finding algo-1 and 2 as functions
1. All updates are in Lines class
1. After call of each line finding/update call display function
1. Write a separate display function that plots either original or binary, warped or unwarped image with estimated lane lines.
1. Write function that measures curvature and offset from lane center. Again this can be called to compute these parameters for most recent lane poly fit values.
1. Write separate Sanity check function that verifies latest estimates of lane lines from Lines class and either udpates or discards the current estimate.
1. Write a function to mapout the current estimates on original unwarped image for video pipeline.
---