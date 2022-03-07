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


  # def skeletonize(self, img):
  #   import si.morphology as m
  #   h1 = np.array([[0, 0, 0],[0, 1, 0],[1, 1, 1]])
  #   m1 = np.array([[1, 1, 1],[0, 0, 0],[0, 0, 0]])
  #   h2 = np.array([[0, 0, 0],[1, 1, 0],[0, 1, 0]])
  #   m2 = np.array([[0, 1, 1],[0, 0, 1],[0, 0, 0]])
  #   hit_list = []
  #   miss_list = []
  #   for k in range(4):
  #       hit_list.append(np.rot90(h1, k))
  #       hit_list.append(np.rot90(h2, k))
  #       miss_list.append(np.rot90(m1, k))
  #       miss_list.append(np.rot90(m2, k))
  #   img = img.copy()
  #   while True:
  #       last = img
  #       for hit, miss in zip(hit_list, miss_list):
  #           hm = m.binary_hit_or_miss(img, hit, miss)
  #           img = np.logical_and(img, np.logical_not(hm))
  #       if np.all(img == last):
  #           break
  #   return img
