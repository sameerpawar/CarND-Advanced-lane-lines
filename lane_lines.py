import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



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

    C = np.absolute(C).astype(float)
    C = np.uint8(255*C/np.max(C))
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

    C = np.absolute(C).astype(float)
    C = np.uint8(255*C/np.max(C))
    binary_output = np.zeros_like(C) 
    binary_output[(thresh[0] < C) & (C <= thresh[1])] = 1
    return binary_output    

# Edit this function to create your own pipeline.
def pipeline(img, ksize = 3):
    image = np.copy(img)
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(10, 100), nchannels = 3)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(30, 100), nchannels = 3)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 255), nchannels = 3)
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(0, 1.4), nchannels = 3)
    s_binary = hls_select(image, thresh=(40, 255), channel = 's')
    h_binary = hls_select(image, thresh=(15, 100), channel = 'h')
    r_binary = rgb_select(image, thresh=(150, 255), channel = 'r')
    

    # Combine various binaries 
    combined_binary = np.zeros_like(image[:,:,0])
    combined_binary[gradx == 1] = 1
    combined_binary[grady == 1] = 1
    combined_binary[h_binary == 1] = 1
    combined_binary[r_binary == 1] = 1
    combined_binary[s_binary == 1] = 1
    combined_binary[mag_binary == 0] = 0
    combined_binary[dir_binary == 0] = 0
    return combined_binary
    
def corners_unwarp(img, nx, ny, mtx, dist):
    # Pass in your image into this function
    # Write code to do the following steps
    # 1) Undistort using mtx and dist
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # 2) Convert to grayscale
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    # 3) Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    img_size = (gray.shape[1], gray.shape[0])
    # 4) If corners found:
    if ret == True:
            # b) define 4 source points src = np.float32([[,],[,],[,],[,]])
                 #Note: you could pick any four of the detected corners 
                 # as long as those four corners define a rectangle
                 #One especially smart way to do this would be to use four well-chosen
                 # corners that were automatically detected during the undistortion steps
                 #We recommend using the automatic detection of corners in your code
        src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
        pt1 = np.min(corners[0])
        print(pt1)
        offset = pt1
        dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
            # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
            # e) use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(undist, M, img_size)
    return warped, M



# Read in an image and grayscale it
image   = mpimg.imread('test6.jpg')
# top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
combined_binary = pipeline(image, ksize = 3)

#plt.imshow(combined_binary)



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(combined_binary, cmap='gray')
ax2.set_title('Combined Thresholded', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()