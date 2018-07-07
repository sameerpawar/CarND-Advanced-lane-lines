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



