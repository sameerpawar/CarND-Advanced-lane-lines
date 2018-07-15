'''Cell-3: Helper functions
1. abs_sobel_thresh
2. mag_thresh
3. dir_threshold
4. hls_select
5. rgb_select
6. get_binary_image
7. Line class
    - set_curvature
    - set_car_offset
    - sanity_check [later]   
    - find_lane_pixels
    - update_lanes
    - display_lanes
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# meters per pixel in X-direction    
mpx_x = 3.8/780
# meters per pixel in Y-direction    
mpx_y = 30/720

def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255), nchannels = 1):
    # Convert to grayscale if the image is color
    if nchannels == 3:
        image   = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if orient == 'x':
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    else:
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    abs_sobel       = np.absolute(sobel)    
    scaled_sobel    = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    grad_binary     = np.zeros_like(scaled_sobel)
    grad_binary[(thresh[0] <= scaled_sobel) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255), nchannels = 1):
    # Convert to grayscale if the image is color
    if nchannels == 3:
        image   = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    abs_sobel       = np.sqrt(np.square(sobelx) + np.square(sobely))    
    scaled_sobel    = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    mag_binary  = np.zeros_like(scaled_sobel)
    mag_binary[(mag_thresh[0] <= scaled_sobel) & (scaled_sobel <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2), nchannels = 1):
    # Convert to grayscale if the image is color
    if nchannels == 3:
        image   = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    abs_sobelx = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = sobel_kernel))
    abs_sobely = np.absolute(cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = sobel_kernel))
    
    grad_direction  = np.arctan2(abs_sobely, abs_sobelx)  
    dir_binary      = np.zeros_like(grad_direction)
    dir_binary[(thresh[0] <= grad_direction) & (grad_direction <= thresh[1])] = 1
    return dir_binary

def hls_select(img, thresh=(0, 255), channel = 's'):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == 'h':
        C = hls[:,:,0]
    if channel == 'l':    
        C = hls[:,:,1]
    if channel == 's':
        C = hls[:,:,2]

    #C = np.absolute(C).astype(float)
    #C = np.uint8(255*C/np.max(C))
    binary_output = np.zeros_like(C) 
    binary_output[(thresh[0] < C) & (C <= thresh[1])] = 1
    return binary_output

def rgb_select(img, thresh=(0, 255), channel = 'r'):
    if channel == 'r':
        C = img[:,:,0]
    if channel == 'g':    
        C = img[:,:,1]
    if channel == 'b':
        C = img[:,:,2]

    #C = np.absolute(C).astype(float)
    #C = np.uint8(255*C/np.max(C))
    binary_output = np.zeros_like(C) 
    binary_output[(thresh[0] < C) & (C <= thresh[1])] = 1
    return binary_output    

def get_binary_image(img):
    image = cv2.undistort(img, mtx, dist, None, mtx)
    kernel_size_gradxy_mag = 7
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=kernel_size_gradxy_mag, thresh=(20, 100), nchannels = 3)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=kernel_size_gradxy_mag, thresh=(30, 100), nchannels = 3)
    mag_binary = mag_thresh(image, sobel_kernel=kernel_size_gradxy_mag, mag_thresh=(30, 255), nchannels = 3)
    dir_binary = dir_threshold(image, sobel_kernel=15, thresh=(0, 1.3), nchannels = 3)
    s_binary = hls_select(image, thresh=(90, 255), channel = 's')
    h_binary = hls_select(image, thresh=(15, 100), channel = 'h')
    r_binary = rgb_select(image, thresh=(150, 255), channel = 'r')
    

    # Combine various binaries 
    combined_binary = np.zeros_like(image[:,:,0])
    combined_binary[gradx == 1] = 1
    combined_binary[grady == 1] = 1
    combined_binary[mag_binary == 0] = 0    
    combined_binary[dir_binary == 0] = 0
    #combined_binary[h_binary == 1] = 1
    #combined_binary[r_binary == 1] = 1
    combined_binary[s_binary == 1] = 1      
    return combined_binary   

# Define a class to receive the characteristics of each line detection
from collections import deque
class Line():
    def __init__(self, side):
        self.side = side
        # was the line detected in the last iteration?
        self.detected = False  
        self.nFailedFrames = 0  
        self.nresetFrames = 0
        # confidence in percentage of range over which search the x-values in hist
        # self.confidence = 0
        # x values of the last n = 5 fits of the line
        # Sam: we only have x-values, y-values are not same in different frames
        #self.recent_xfitted = [] 
        #average x values of the fitted line over the last n = 5 iterations
        # Sam: we only have x-values, y-values are not same in different frames
        #self.bestx = None     
        #polynomial coefficients averaged over the last n = 5 iterations
        self.best_fit = None        
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = 0 
        #difference in fit coefficients between last and new fits
        #self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        #self.allx = None  
        #y values for detected line pixels
        #self.ally = None

    def update_curvature(self):
        # computes and updates the curvature for best fit
        # change it from pixels to meters
        a, b, c = self.best_fit
        a = a*mpx_x/(mpx_y**2)
        b = b*mpx_x/mpx_y
        self.radius_of_curvature = ((1 + (2*a*30 + b)**2)**1.5) / np.absolute(2*a)

    def set_detected(self):
        # Good lane detected
        self.detected = True
        self.nFailedFrames = 0

    def updateCounter(self):
        self.nFailedFrames += 1
        if self.nFailedFrames > 10:
            self.detected = False
            self.nFailedFrames = 0
        
    def reset_detected(self):
        # computes and updates the curvature for current fit
        self.detected = False
        
    def update_best_fit(self):
        self.best_fit = self.best_fit*0.7 + 0.3*self.current_fit
        
    def update_line_base_pos(self):
        # computes and updates the curvature for current fit
        self.line_base_pos = 0

    def update_lane_line(self, binary_image, alg = 'alg1'):            
        
        #******************************************************************************
        # Set the hyper parameters and initialization
        #******************************************************************************
        FIT2WICE = True
        filter_width = 10
        window_filter = np.ones(filter_width)    
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50        
        # Choose the number of sliding windows for finding pixels
        nwindows = 9
        binary_warped = cv2.warpPerspective(binary_image, M, (binary_image.shape[1], binary_image.shape[0]), flags=cv2.INTER_NEAREST)  # keep same size as input image            
        window_height = np.int(binary_warped.shape[0]//nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        #******************************************************************************
        
        if self.detected:
            # If previous lane was detected with great confidence
            margin = 60
            lane_fit = self.best_fit
            lane_inds = ((nonzerox > (lane_fit[0]*(nonzeroy**2) + lane_fit[1]*nonzeroy + 
            lane_fit[2] - margin)) & (nonzerox < (lane_fit[0]*(nonzeroy**2) + 
            lane_fit[1]*nonzeroy + lane_fit[2] + margin)))

            # Again, extract lane pixel positions
            lanex = nonzerox[lane_inds]
            laney = nonzeroy[lane_inds] 
        else:
            # [sam] Instead of argmax over all the range. This can be modified to look around some estimate from history
            # maintaining history needs to be thought.            
            # Take a histogram of the bottom half of the image
            if self.side == 'left':
                l_sum   = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:int(binary_warped.shape[1]/2)], axis=0)
                x_base  = np.argmax(np.convolve(window_filter,l_sum))-filter_width/2
            else:
                r_sum   = np.sum(binary_warped[int(binary_warped.shape[0]/2):,int(binary_warped.shape[1]/2):], axis=0)
                x_base  = np.argmax(np.convolve(window_filter,r_sum))-filter_width/2 + int(binary_warped.shape[1]/2)

            # Current positions to be updated for each window
            x_current = x_base
            # Create empty list to receive lane pixel indices
            lane_inds = []           

            # Step through the windows one by one
            for window in range(nwindows):
                # Identify window boundaries in x and y (and right and left)
                win_y_low   = binary_warped.shape[0] - (window+1)*window_height
                win_y_high  = binary_warped.shape[0] - window*window_height
                win_x_low   = x_current - margin
                win_x_high  = x_current + margin
                # Identify the nonzero pixels in x and y within the window
                good_lane_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]
                # Append these indices to the lists
                lane_inds.append(good_lane_inds)
                # If you found > minpix pixels, recenter next window on their mean position
                if len(good_lane_inds) > minpix:
                    x_current = np.int(np.mean(nonzerox[good_lane_inds]))
            # Concatenate the arrays of indices
            lane_inds  = np.concatenate(lane_inds)

            # Extract left and right line pixel positions
            lanex = nonzerox[lane_inds]
            laney = nonzeroy[lane_inds] 

        # Fit a second order polynomial to each
        lane_fit = np.polyfit(laney, lanex, 2)

        
        if FIT2WICE:
            margin1 = 50
            lane_inds = ((nonzerox > (lane_fit[0]*(nonzeroy**2) + lane_fit[1]*nonzeroy + 
            lane_fit[2] - margin1)) & (nonzerox < (lane_fit[0]*(nonzeroy**2) + 
            lane_fit[1]*nonzeroy + lane_fit[2] + margin1)))

            # Again, extract lane pixel positions
            lanex = nonzerox[lane_inds]
            laney = nonzeroy[lane_inds] 

        # Fit a second order polynomial to each
        self.current_fit = np.polyfit(laney, lanex, 2)
        if self.best_fit is None:
            self.best_fit = self.current_fit
        return

def sanity_check(left_lane, right_lane):    
    sanity_flag = True
    # compute lane width in meters for current fit
    la, lb, lc = left_lane.current_fit
    ra, rb, rc = right_lane.current_fit
    y = np.linspace(0, 720, 121)
    left_lane_pts = (la*(y**2) + lb*y + lc)
    right_lane_pts = (ra*(y**2) + rb*y + rc)
    diff = np.absolute(left_lane_pts-right_lane_pts)
    # lane width in meters
    lane_width = np.mean(diff)*mpx_x
    
    # Check if lanes are almost parallel
    if (np.max(diff)/np.min(diff)) > 1.15:
           sanity_flag = False
    
    # check if lane width makes sense
    if lane_width < 3 or lane_width > 5:
        sanity_flag = False
    
    if sanity_flag:
        # Adds to the history of fits
        # sets the detected flag
        left_lane.update_best_fit()
        left_lane.set_detected()
        right_lane.update_best_fit()
        right_lane.set_detected()
    else:
        #increase failed frame counter
        left_lane.updateCounter()
        right_lane.updateCounter()        
        if left_lane.nFailedFrames == 0:
            left_lane.nresetFrames += 1
            print("reset = ", left_lane.nresetFrames)
        
def display_lanes(image, left_lane, right_lane, style = 'orig'):
    left_fit = left_lane.best_fit
    right_fit = right_lane.best_fit
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw on and an image to show the selection window
    color_warp = np.zeros_like(image)
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    if style == 'orig':
        unwarped_image = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
        result = cv2.addWeighted(image, 1, unwarped_image, 0.3, 0) 
    else:
        unwarped_image = color_warp
        image  = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
        result = cv2.addWeighted(image, 1, unwarped_image, 0.3, 0)     
    
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText1 = (100,50)
    bottomLeftCornerOfText2 = (100,100)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    
    # compute the lane curvature and offset from center
    right_lane.update_curvature()
    radius = np.round(right_lane.radius_of_curvature*100)/100
    # compute lane offset from center of image
    la, lb, lc = left_lane.best_fit
    ra, rb, rc = right_lane.best_fit
    y = np.linspace(0, 720, 121)
    left_lane_pts = (la*(y**2) + lb*y + lc)
    right_lane_pts = (ra*(y**2) + rb*y + rc)
    lane_width_pixels = np.absolute(right_lane_pts-left_lane_pts)
    lane_array_str = str(np.array([np.max(lane_width_pixels)/np.min(lane_width_pixels), np.mean(lane_width_pixels), np.max(lane_width_pixels)]))
    
    center = lane_width_pixels[-1]
    offset_pixels = 1280/2-center
    offset_meters = np.round(offset_pixels*mpx_x*100)/100

    if offset_meters < 0:
        offset_str = "Vehicle is " + str(-offset_meters) + "(m) right of the center"
    else:
        offset_str = "Vehicle is " + str(offset_meters) + "(m) left of the center"
     
    
    cv2.putText(result,'Radius of curvature = ' + str(radius) + "(m)", 
        bottomLeftCornerOfText1, 
        font, 
        fontScale,
        fontColor,
        lineType)

    cv2.putText(result,offset_str, 
        bottomLeftCornerOfText2, 
        font, 
        fontScale,
        fontColor,
        lineType)

    
    
    return result
