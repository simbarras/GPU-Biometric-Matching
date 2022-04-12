"""
" Other feature extraction methods from IDIAP's bob library https://www.idiap.ch/software/bob/
"""

import logging
import numpy as np
import scipy.ndimage as si
import time
from PIL import Image
import math
import cv2
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)

def wide_line_detector(image,mask):
    """Computes and returns the Wide Line Detector features for the given input
    fingervein image
    Based on B. Huang, Y. Dai, R. Li, D. Tang and W. Li. Finger-vein
    authentication based on wide line detector and pattern normalization,
    Proceedings on 20th International Conference on Pattern Recognition (ICPR),
    2010."""

    radius = 5    #Radius of the circular neighbourhood region
    threshold = 1 #Neigborhood threshold
    g = 41         #Sum of neigbourhood threshold
    rescale = False #was originally true yet have some issues with the scale function, could not import and what I am doing idk if right
    finger_image = image.astype(np.float64)

    finger_mask = np.zeros(mask.shape)
    finger_mask[mask == True] = 1

    # Rescale image if required
    if rescale == True:
      scaling_factor = 0.24
      # finger_image = sm.imresize(finger_image,scaling_factor)#.astype()
      finger_image = np.array(Image.fromarray(finger_image).resize((int(finger_image.shape[1] * scaling_factor),int(finger_image.shape[0]*scaling_factor)),Image.BILINEAR))
      finger_mask = np.array(Image.fromarray(finger_mask).resize((int(finger_mask.shape[1]*scaling_factor),int(finger_mask.shape[0]*scaling_factor)),Image.BILINEAR))
      # finger_image = scale(finger_image,scaling_factor)
      # finger_mask = sm.imresize(finger_mask,scaling_factor)
      # finger_mask = scale(finger_mask,scaling_factor)
      #To eliminate residuals from the scalation of the binary mask
      finger_mask = si.binary_dilation(finger_mask, structure=np.ones((1,1))).astype(int)

    x = np.arange((-1)*radius, radius+1)
    y = np.arange((-1)*radius, radius+1)
    X, Y = np.meshgrid(x,y)

    N = X**2 + Y**2 <= radius**2  # Neighbourhood mask

    img_h, img_w = finger_image.shape  #Image height and width

    veins = np.zeros(finger_image.shape)

    for y in range(radius,img_h-radius):
        for x in range(radius,img_w-radius):
            s=((finger_image[y-radius:y+radius+1,x-radius:x+radius+1] - finger_image[y,x]) <= threshold)
            m = (s*N).sum()
            veins[y,x] = float(m <= g)

    # Mask the vein image with the finger region
    img_veins_bin = veins*finger_mask

    return img_veins_bin

def repeated_line_tracking(finger_image, mask):
    """Computes and returns the MiuraMax features for the given input
    fingervein image
    Based on N. Miura, A. Nagasaka, and T. Miyatake. Feature extraction of finger
    vein patterns based on repeated line tracking and its application to personal
    identification. Machine Vision and Applications, Vol. 15, Num. 4, pp.
    194--203, 2004"""
    iterations = 3000 # Maximum number of iterations
    r = 1             # Distance between tracking point and cross section of the profile
    profile_w = 21    # Width of profile (Error: profile_w must be odd)
    rescale = False #this was set to true, need to figure it out,experimental resize currently
    seed = 0

    # Sets the random seed before starting to process
    np.random.seed(seed)

    finger_mask = np.zeros(mask.shape)
    finger_mask[mask == True] = 1

    # Rescale image if required
    if rescale == True:
      scaling_factor = 0.6

      # finger_image = bob.ip.base.scale(finger_image,scaling_factor)
      # finger_mask = bob.ip.base.scale(finger_mask,scaling_factor)
      finger_image = np.array(Image.fromarray(finger_image).resize((int(finger_image.shape[1] * scaling_factor),int(finger_image.shape[0]*scaling_factor)),Image.BILINEAR))
      finger_mask = np.array(Image.fromarray(finger_mask).resize((int(finger_mask.shape[1]*scaling_factor),int(finger_mask.shape[0]*scaling_factor)),Image.BILINEAR))


      #To eliminate residuals from the scalation of the binary mask
      finger_mask = si.binary_dilation(finger_mask, structure=np.ones((1,1))).astype(int)

    p_lr = 0.5  # Probability of goin left or right
    p_ud = 0.25 # Probability of going up or down

    Tr = np.zeros(finger_image.shape) # Locus space
    filtermask = np.array(([-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]))

    # Check if progile w is even
    if (profile_w.__mod__(2) == 0):
        print ('Error: profile_w must be odd')

    ro = np.round(r*math.sqrt(2)/2)    # r for oblique directions
    hW = (profile_w-1)/2                  # half width for horz. and vert. directions
    hWo = np.round(hW*math.sqrt(2)/2)       # half width for oblique directions

    # Omit unreachable borders
    border = int(r+hW)
    finger_mask[0:border,:] = 0
    finger_mask[finger_mask.shape[0]-border:,:] = 0
    finger_mask[:,0:border] = 0
    finger_mask[:,finger_mask.shape[1]-border:] = 0

    ## Uniformly distributed starting points
    aux = np.argwhere( (finger_mask > 0) == True )
    indices = np.random.permutation(aux)
    indices = indices[0:iterations,:]    # Limit to number of iterations

    ## Iterate through all starting points
    for it in range(0,iterations):
        yc = indices[it,0] # Current tracking point, y
        xc = indices[it,1] # Current tracking point, x

        # Determine the moving-direction attributes
        # Going left or right ?
        if (np.random.random_sample() >= 0.5):
            Dlr = -1  # Going left
        else:
            Dlr = 1   # Going right

        # Going up or down ?
        if (np.random.random_sample() >= 0.5):
            Dud = -1  # Going up
        else:
            Dud = 1   # Going down

        # Initialize locus-positition table Tc
        Tc = np.zeros(finger_image.shape, np.bool)

        #Dlr = -1; Dud=-1; LET OP
        Vl = 1
        while (Vl > 0):
            # Determine the moving candidate point set Nc
            Nr = np.zeros([3,3], np.bool)
            Rnd = np.random.random_sample()
            #Rnd = 0.8 LET OP
            if (Rnd < p_lr):
                # Going left or right
                Nr[:,1+Dlr] = True
            elif (Rnd >= p_lr) and (Rnd < (p_lr + p_ud)):
                # Going up or down
                Nr[1+Dud,:] = True
            else:
                # Going any direction
                Nr = np.ones([3,3], np.bool)
                Nr[1,1] = False
            #tmp = np.argwhere( (~Tc[yc-2:yc+1,xc-2:xc+1] & Nr & finger_mask[yc-2:yc+1,xc-2:xc+1].astype(np.bool)).T.reshape(-1) == True )
            tmp = np.argwhere( (~Tc[yc-1:yc+2,xc-1:xc+2] & Nr & finger_mask[yc-1:yc+2,xc-1:xc+2].astype(np.bool)).T.reshape(-1) == True )
            Nc = np.concatenate((xc + filtermask[tmp,0],yc + filtermask[tmp,1]),axis=1)
            if (Nc.size==0):
                Vl=-1
                continue

            ## Detect dark line direction near current tracking point
            Vdepths = np.zeros((Nc.shape[0],1)) # Valley depths
            for i in range(0,Nc.shape[0]):
                ## Horizontal or vertical
                if (Nc[i,1] == yc):
                    # Horizontal plane
                    yp = Nc[i,1]
                    if (Nc[i,0] > xc):
                        # Right direction
                        xp = Nc[i,0] + r
                    else:
                        # Left direction
                        xp = Nc[i,0] - r
                    Vdepths[i] = finger_image[int(yp + hW), int(xp)] - 2*finger_image[int(yp),int(xp)] + finger_image[int(yp - hW), int(xp)]
                elif (Nc[i,0] == xc):
                    # Vertical plane
                    xp = Nc[i,0]
                    if (Nc[i,1] > yc):
                        # Down direction
                        yp = Nc[i,1] + r
                    else:
                        # Up direction
                        yp = Nc[i,1] - r
                    Vdepths[i] = finger_image[int(yp), int(xp + hW)] - 2*finger_image[int(yp),int(xp)] + finger_image[int(yp), int(xp - hW)]

                ## Oblique directions
                if ( (Nc[i,0] > xc) and (Nc[i,1] < yc) ) or ( (Nc[i,0] < xc) and (Nc[i,1] > yc) ):
                    # Diagonal, up /
                    if (Nc[i,0] > xc and Nc[i,1] < yc):
                        # Top right
                        xp = Nc[i,0] + ro
                        yp = Nc[i,1] - ro
                    else:
                        # Bottom left
                        xp = Nc[i,0] - ro
                        yp = Nc[i,1] + ro
                    Vdepths[i] = finger_image[int(yp - hWo), int(xp - hWo)] - 2*finger_image[int(yp),int(xp)] + finger_image[int(yp + hWo), int(xp + hWo)]
                else:
                    # Diagonal, down \
                    if (Nc[i,0] < xc and Nc[i,1] < yc):
                        # Top left
                        xp = Nc[i,0] - ro
                        yp = Nc[i,1] - ro
                    else:
                        # Bottom right
                        xp = Nc[i,0] + ro
                        yp = Nc[i,1] + ro
                    Vdepths[i] = finger_image[int(yp + hWo), int(xp - hWo)] - 2*finger_image[int(yp),int(xp)] + finger_image[int(yp - hWo), int(xp + hWo)]
            # End search of candidates
            index = np.argmax(Vdepths)  #Determine best candidate
            # Register tracking information
            Tc[yc, xc] = True
            # Increase value of tracking space
            Tr[yc, xc] = Tr[yc, xc] + 1
            # Move tracking point
            xc = Nc[index, 0]
            yc = Nc[index, 1]

    img_veins = Tr

    # Binarise the vein image
    md = np.median(img_veins[img_veins>0])
    img_veins_bin = img_veins > md
    img_veins_bin = si.binary_closing(img_veins_bin, structure=np.ones((2,2))).astype(int)

    return img_veins_bin.astype(np.float64)


##################### Maximum Curvature Algorithm

def detect_valleys(image, mask, sigma):

    """ Detects valleys on the image respecting the mask

    This step corresponds to Step 1-1 in the original paper. The objective is,
    for all 4 cross-sections (z) of the image (horizontal, vertical, 45 and -45
    diagonals), to compute the following proposed valley detector as defined in
    Equation 1, page 348:

    .. math::

       \kappa(z) = \\frac{d^2P_f(z)/dz^2}{(1 + (dP_f(z)/dz)^2)^\\frac{3}{2}}


    We start the algorithm by smoothing the image with a 2-dimensional gaussian
    filter. The equation that defines the kernel for the filter is:

    .. math::

       \mathcal{N}(x,y)=\\frac{1}{2\pi\sigma^2}e^\\frac{-(x^2+y^2)}{2\sigma^2}


    This is done to avoid noise from the raw data (from the sensor). The
    maximum curvature method then requires we compute the first and second
    derivative of the image for all cross-sections, as per the equation above.

    We instead take the following equivalent approach:

    1. construct a gaussian filter
    2. take the first (dh/dx) and second (d^2/dh^2) deritivatives of the filter
    3. calculate the first and second derivatives of the smoothed signal using
       the results from 3. This is done for all directions we're interested in:
       horizontal, vertical and 2 diagonals. First and second derivatives of a
       convolved signal

    .. note::

       Item 3 above is only possible thanks to the steerable filter property of
       the gaussian kernel. See "The Design and Use of Steerable Filters" from
       Freeman and Adelson, IEEE Transactions on Pattern Analysis and Machine
       Intelligence, Vol. 13, No. 9, September 1991.


    @param image (numpy.ndarray) : an array of 64-bit floats containing the input image
    @param mask (numpy.ndarray) : an array, of the same size as ``image``, containing a mask (booleans) indicating where the finger is on ``image``.
    @param sigma (float) : Variance of the gaussian filter

    @return (numpy.ndarray) : a 3-dimensional array of 64-bits containing $\kappa$ for
        all considered directions. $\kappa$ has the same shape as ``image``,
        except for the 3rd. dimension, which provides planes for the
        cross-section valley detections for each of the contemplated directions,
        in this order: horizontal, vertical, +45 degrees, -45 degrees.

    """

    # 1. constructs the 2D gaussian filter "h" given the window size,
    # extrapolated from the "sigma" parameter (4x)
    # N.B.: This is a text-book gaussian filter definition
    winsize = np.ceil(4 * sigma) # enough space for the filter
    window = np.arange(-winsize, winsize + 1)
    X, Y = np.meshgrid(window, window)
    G = 1.0 / (2 * np.pi * sigma ** 2)
    G *= np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))

    # 2. calculates first and second derivatives of "G" with respect to "X"
    # (0), "Y" (90 degrees) and 45 degrees (?)
    G1_0 = (-X / (sigma ** 2)) * G
    G2_0 = ((X ** 2 - sigma ** 2) / (sigma ** 4)) * G
    G1_90 = G1_0.T
    G2_90 = G2_0.T
    hxy = ((X * Y) / (sigma ** 4)) * G

    # 3. calculates derivatives w.r.t. to all directions of interest
    #    stores results in the variable "k". The entries (last dimension) in k
    #    correspond to curvature detectors in the following directions:
    #
    #    [0] horizontal
    #    [1] vertical
    #    [2] diagonal \ (45 degrees rotation)
    #    [3] diagonal / (-45 degrees rotation)
    image_g1_0 = si.convolve(image, G1_0, mode = 'nearest')
    image_g2_0 = si.convolve(image, G2_0, mode = 'nearest')
    image_g1_90 = si.convolve(image, G1_90, mode = 'nearest')
    image_g2_90 = si.convolve(image, G2_90, mode = 'nearest')
    fxy = si.convolve(image, hxy, mode = 'nearest')

    # support calculation for diagonals, given the gaussian kernel is
    # steerable. To calculate the derivatives for the "\" diagonal, we first
    # **would** have to rotate the image 45 degrees counter-clockwise (so the
    # diagonal lies on the horizontal axis). Using the steerable property, we
    # can evaluate the first derivative like this:
    #
    # image_g1_45 = cos(45)*image_g1_0 + sin(45)*image_g1_90
    #             = sqrt(2)/2*fx + sqrt(2)/2*fx
    #
    # to calculate the first derivative for the "/" diagonal, we first
    # **would** have to rotate the image -45 degrees "counter"-clockwise.
    # Therefore, we can calculate it like this:
    #
    # image_g1_m45 = cos(-45)*image_g1_0 + sin(-45)*image_g1_90
    #              = sqrt(2)/2*image_g1_0 - sqrt(2)/2*image_g1_90
    #

    image_g1_45 = 0.5 * np.sqrt(2) * (image_g1_0 + image_g1_90)
    image_g1_m45 = 0.5 * np.sqrt(2) * (image_g1_0 - image_g1_90)

    # NOTE: You can't really get image_g2_45 and image_g2_m45 from the theory
    # of steerable filters. In contact with B.Ton, he suggested the following
    # material, where that is explained: Chapter 5.2.3 of van der Heijden, F.
    # (1994) Image based measurement systems: object recognition and parameter
    # estimation. John Wiley & Sons Ltd, Chichester. ISBN 978-0-471-95062-2

    # This also shows the same result:
    # http://www.mif.vu.lt/atpazinimas/dip/FIP/fip-Derivati.html (look for
    # SDGD)

    # He also suggested to look at slide 75 of the following presentation
    # indicating it is self-explanatory: http://slideplayer.com/slide/5084635/

    image_g2_45 = 0.5 * image_g2_0 + fxy + 0.5 * image_g2_90
    image_g2_m45 = 0.5 * image_g2_0 - fxy + 0.5 * image_g2_90

    # ######################################################################
    # [Step 1-1] Calculation of curvature profiles
    # ######################################################################

    # Peak detection (k or kappa) calculation as per equation (1) page 348 on
    # Miura's paper
    finger_mask = mask.astype('float64')

    return np.dstack([
        (image_g2_0 / ((1 + image_g1_0 ** 2) ** (1.5))) * finger_mask,
        (image_g2_90 / ((1 + image_g1_90 ** 2) ** (1.5))) * finger_mask,
        (image_g2_45 / ((1 + image_g1_45 ** 2) ** (1.5))) * finger_mask,
        (image_g2_m45 / ((1 + image_g1_m45 ** 2) ** (1.5))) * finger_mask,
    ])

def eval_vein_probabilities(k):

    """ Evaluates joint vein centre probabilities from cross-sections

    This function will take $\kappa$ and will calculate the vein centre
    probabilities taking into consideration valley widths and depths. It
    aggregates the following steps from the paper:

    * [Step 1-2] Detection of the centres of veins
    * [Step 1-3] Assignment of scores to the centre positions
    * [Step 1-4] Calculation of all the profiles

    Once the arrays of curvatures (concavities) are calculated, here is how
    detection works: The code scans the image in a precise direction (vertical,
    horizontal, diagonal, etc). It tries to find a concavity on that direction
    and measure its width (see Wr on Figure 3 on the original paper). It then
    identifies the centers of the concavity and assign a value to it, which
    depends on its width (Wr) and maximum depth (where the peak of darkness
    occurs) in such a concavity. This value is accumulated on a variable (Vt),
    which is re-used for all directions. Vt represents the vein probabilites
    from the paper.


    @param k (numpy.ndarray): a 3-dimensional array of 64-bits containing $\kappa$
        for all considered directions. $\kappa$ has the same shape as
        ``image``, except for the 3rd. dimension, which provides planes for the
        cross-section valley detections for each of the contemplated
        directions, in this order: horizontal, vertical, +45 degrees, -45
        degrees.

    @return (numpy.ndarray): The un-accumulated vein centre probabilities ``V``. This
        is a 3D array with 64-bit floats with the same dimensions of the input
        array ``k``. You must accumulate (sum) over the last dimension to
        retrieve the variable ``V`` from the paper.
    """

    V = np.zeros(k.shape[:2], dtype = 'float64')


    def _prob_1d(a):

        """ Finds "vein probabilities" in a 1-D signal

        This function efficiently counts the width and height of concavities in
        the cross-section (1-D) curvature signal ``s``.

        It works like this:

        1. We create a 1-shift difference between the thresholded signal and itself
        2. We compensate for starting and ending regions
        3. For each sequence of start/ends, we compute the maximum in the original signal

        Example (mixed with pseudo-code):

        a = 0 1 2 3 2 1 0 -1 0 0 1 2 5 2 2 2 1
        b = a > 0 (as type int)
        b = 0 1 1 1 1 1 0  0 0 0 1 1 1 1 1 1 1

        0 1 1 1 1 1  0 0 0 0 1 1 1 1 1 1 1
          0 1 1 1 1  1 0 0 0 0 1 1 1 1 1 1 1 (-)
        -------------------------------------------
        X 1 0 0 0 0 -1 0 0 0 1 0 0 0 0 0 0 X (length is smaller than orig.)

        starts = numpy.where(diff > 0)
        ends   = numpy.where(diff < 0)

        -> now the number of starts and ends should match, otherwise, we must compensate

            -> case 1: b starts with 1: add one start in begin of "starts"
            -> case 2: b ends with 1: add one end in the end of "ends"

        -> iterate over the sequence of starts/ends and find maximums

        @param a (numpy.ndarray): 1D signal with curvature to explore

        @return (numpy.ndarray): 1D container with the vein centre probabilities
        """

        b = (a > 0).astype(int)
        diff = b[1:] - b[:-1]
        starts = np.argwhere(diff > 0)
        starts += 1 # compensates for shifted different
        ends = np.argwhere(diff < 0)
        ends += 1 # compensates for shifted different
        if b[0]:
            starts = np.insert(starts, 0, 0)
        if b[-1]:
            ends = np.append(ends, len(a))

        z = np.zeros_like(a)

        if starts.size == 0 and ends.size == 0:
            return z

        for start, end in zip(starts, ends):
            maximum = np.argmax(a[int(start):int(end)])
            z[start + maximum] = a[start + maximum] * (end - start)

        return z


    # Horizontal direction
    for index in range(k.shape[0]):
        V[index, :] += _prob_1d(k[index, :, 0])

    # Vertical direction
    for index in range(k.shape[1]):
        V[:, index] += _prob_1d(k[:, index, 1])

    # Direction: 45 degrees (\)
    curv = k[:, :, 2]
    i, j = np.indices(curv.shape)
    for index in range(-curv.shape[0] + 1, curv.shape[1]):
        V[i == (j - index)] += _prob_1d(curv.diagonal(index))

    # Direction: -45 degrees (/)
    # NOTE: due to the way the access to the diagonals are implemented, in this
    # loop, we operate bottom-up. To match this behaviour, we also address V
    # through Vud.
    # required so we get "/" diagonals correctly
    curv = np.flipud(k[:, :, 3])
    Vud = np.flipud(V) # match above inversion
    for index in reversed(range(curv.shape[1] - 1, -curv.shape[0], -1)):
        Vud[i == (j - index)] += _prob_1d(curv.diagonal(index))

    return V

def connect_centres(V):

    """ Connects vein centres by filtering vein probabilities ``V``

    This function does the equivalent of Step 2 / Equation 4 at Miura's paper.

    The operation is applied on a row from the ``V`` matrix, which may be
    acquired horizontally, vertically or on a diagonal direction. The pixel
    value is then reset in the center of a windowing operation (width = 5) with
    the following value:

    .. math::

        b[w] = min(max(a[w+1], a[w+2]) + max(a[w-1], a[w-2]))


    @param V (numpy.ndarray): The accumulated vein centre probabilities ``V``. This
        is a 2D array with 64-bit floats and is defined by Equation (3) on the
        paper.

    @return (numpy.ndarray): A 3-dimensional 64-bit array ``Cd`` containing the result
        of the filtering operation for each of the directions. ``Cd`` has the
        dimensions of $\kappa$ and $V_i$. Each of the planes correspond to the
        horizontal, vertical, +45 and -45 directions.
    """


    def _connect_1d(a):

        """ Connects centres in the given vector

        The strategy we use to vectorize this is to shift a twice to the left and
        twice to the right and apply a vectorized operation to compute the above.

        @param a (numpy.ndarray): Input 1D array which will be window scanned

        @return numpy.ndarray: Output 1D array (must be writable), in which we will
            set the corrected pixel values after the filtering above. Notice that,
            given the windowing operation, the returned array size would be 4 short
            of the input array.

        """

        return np.amin([np.amax([a[3:-1], a[4:]], axis = 0), np.amax([a[1:-3], a[:-4]], axis = 0)], axis = 0)


    Cd = np.zeros(V.shape + (4,), dtype = 'float64')

    # Horizontal direction
    for index in range(V.shape[0]):
        Cd[index, 2:-2, 0] = _connect_1d(V[index, :])

    # Vertical direction
    for index in range(V.shape[1]):
        Cd[2:-2, index, 1] = _connect_1d(V[:, index])

    # Direction: 45 degrees (\)
    i, j = np.indices(V.shape)
    border = np.zeros((2,), dtype = 'float64')
    for index in range(-V.shape[0] + 5, V.shape[1] - 4):
        # NOTE: hstack **absolutely** necessary here as double indexing after
        # array indexing is **not** possible with np (it returns a copy)
        Cd[:, :, 2][i == (j - index)] = np.hstack([border, _connect_1d(V.diagonal(index)), border])

    # Direction: -45 degrees (/)
    Vud = np.flipud(V)
    Cdud = np.flipud(Cd[:, :, 3])
    for index in reversed(range(V.shape[1] - 5, -V.shape[0] + 4, -1)):
        # NOTE: hstack **absolutately** necessary here as double indexing after
        # array indexing is **not** possible with np (it returns a copy)
        Cdud[:, :][i == (j - index)] = np.hstack([border, _connect_1d(Vud.diagonal(index)), border])

    return Cd

def binarise(G):

    """ Binarise vein images using a threshold assuming distribution is diphasic

    This function implements Step 3 of the paper. It binarises the 2-D array
    ``G`` assuming its histogram is mostly diphasic and using a median value.

    @param G (numpy.ndarray): A 2-dimensional 64-bit array ``G`` containing the
        result of the filtering operation. ``G`` has the dimensions of the original image.

    @return (numpy.ndarray): A 2-dimensional 64-bit float array with the same
        dimensions of the input image, but containing its vein-binarised version.
        The output of this function corresponds to the output of the method.

    """

    median = np.median(G[G > 0])
    Gbool = G > median
    return Gbool.astype(np.float64)

def maximum_curvature(image, mask, sigma):

    """ Extracts an image (given in the form of a numpy array)'s fingerveins
    The extracting steps were chosen from bob's api to mimic the original
    extract_features script which used bob's library.

    @param image (numpy.ndarray) : The numpy array to extract veins from
    @param mask (numpy.ndarray) : The mask representing the finger
    @param sigma (float) : The sigma parameter for detect_valleys

    @return (numpy.ndarray of float64 0's and 1's) : The extracted image
    """

    finger_image = image.astype('float64')

    start = time.time()
    kappa = detect_valleys(finger_image, mask, sigma)
    log.info(f"filtering took {time.time() - start:.2f} seconds")

    start = time.time()
    V = eval_vein_probabilities(kappa)
    log.info(f"probabilities took {time.time() - start:.2f} seconds")

    start = time.time()
    Cd = connect_centres(V)
    log.info(f"connections took {time.time() - start:.2f} seconds")

    start = time.time()
    retval = binarise(np.amax(Cd, axis = 2))
    log.info(f"binarization took {time.time() - start:.2f} seconds")

    return retval,mask
