"""
" All the background elimination
" routines in one place
"""
import cv2 as cv
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage as si
import scipy.signal as sp
import subprocess
from PIL import Image
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
from skimage.feature import canny
from skimage.morphology import convex_hull_image
from .utils import show_bool, show_uint16, img_hist
from .preprocess import histogram_equalization

log = logging.getLogger(__name__)

#nxn matrix of -1's with an n² - 1 at its center
NMS_FILTER = lambda n : np.array([[-1] * n] * (n//2)
                                +[[-1] * (n//2) + [n**2 - 1] + [-1] * (n//2)]
                                +[[-1] * n] * (n//2))


def partial_segment(img, pt_1, pt_2, fpt_1, fpt_2):
    """
    @param img: original image
    @param pt_1: first point of ROI rectangle (upper left)
    @param pt_2: second point of ROI rectangle (lower right)
    @param fpt_1: fist point of finger region
    @param fpt_2: second point of finger region
    @return: mask of entire image where only masked inside ROI
    """

    # cut out from image and perform hist eq.
    mask = np.zeros_like(img)
    mask[fpt_1[1] : fpt_2[1], fpt_1[0]:fpt_2[0]] = 1
    W = img[pt_1[1] : pt_2[1], pt_1[0]:pt_2[0]]
    plt.imshow(W)
    plt.show()

    return img

def border(np_img, cam):

    if cam == 1:

        # left
        partial_segment(np_img, (20, 60), (80, 200), (50, 120), (80, 140))
        cv.rectangle(np_img, (20, 60), (80, 200), (255, 255, 0))
        cv.rectangle(np_img, (50, 120), (80, 140), (255, 255, 0))

        # right
        cv.rectangle(np_img, (300, 30), (360, 200), (255, 0, 255))
        cv.rectangle(np_img, (300, 105), (330, 125), (255, 255, 0))


    else:
        # right
        cv.rectangle(np_img, (310, 50), (370, 190), (255, 255, 0))
        cv.rectangle(np_img, (310, 110), (340, 130), (255, 255, 0))

        # left
        cv.rectangle(np_img, (40, 30), (100, 200), (255, 0, 255))
        cv.rectangle(np_img, (40, 105), (70, 125), (255, 255, 0))

    plt.imshow(np_img)
    plt.show()

    return np_img

def remove_static_mask(np_img, cam):
    if cam == 1:
        mask_img = Image.open("resources/mask_cam1.png")
    if cam == 2:
        mask_img = Image.open("resources/mask_cam2.png")

    M = np.asarray(mask_img)
    np_img[M[:,:,1] == 255] = 0
    return np_img

def max_thresh(arr, start, dir, threshold):
    val = 0
    idx = start
    prev_val = 0
    max_val = 0
    max_idx = start
    if dir == "up":
        while val < threshold and idx > 30:
            val = arr[idx]
            if val > max_val:
                max_val = val
                max_idx = idx
            idx -= 1
            if prev_val >= threshold and val < prev_val:
                idx = idx + 1
                break
    else:
        while val < threshold and idx < 220:
            val = arr[idx]
            if val > max_val:
                max_val = val
                max_idx = idx
            idx += 1
            if prev_val >= threshold and val < prev_val:
                idx = idx - 1
                break

    if idx == 220 or idx == 30:
        return max_idx

    return idx

def edge_points(img, x_1, f_1, threshold=4):

    avg_1 = np.sum(img[:, x_1 : x_1 + 1], axis=1)
    avg_1 = avg_1 / np.average(avg_1)
    #plt.plot(avg_1)
    #plt.show()
    a = (max_thresh(avg_1, f_1, "up", threshold))
    b = (max_thresh(avg_1, f_1, "down", threshold))
    return [(x_1, a), (x_1, b)]

def edge_mask(img, cam, roi_1=(35, 355), roi_2=(55, 360)):
    #plt.imshow(img)
    #plt.show()
    if cam == 1:
        roi = roi_1
    else:
        roi = roi_2

    gx, gy = np.gradient(img)
    gradient = np.hypot(gx, gy)
    mid_y = 130

    points_up_x, points_up_y, points_down_x, points_down_y = [], [], [], []
    for i in range(*roi):
        ps = edge_points(gradient, i, mid_y)
        points_up_x.append(ps[0][0])
        points_up_y.append(ps[0][1])
        points_down_x.append(ps[1][0])
        points_down_y.append(ps[1][1])

    #plt.plot(points_up_x, points_up_y, color="red")
    #plt.plot(points_down_x, points_down_y, color="orange")
    #plt.imshow(img)
    #plt.show()

    mask = np.zeros_like(img)
    for ux, uy, dx, dy in zip(points_up_x, points_up_y, points_down_x, points_down_y):
        mask[uy : dy, ux] = 1
    #plt.imshow(mask)
    #plt.show()

    # detect and remove outliers:
    points_down_y = si.gaussian_filter1d(points_down_y, 3)
    points_up_y = si.gaussian_filter1d(points_up_y, 3)


    # remove very bright spots
    mask[img > 240] = 0
    #plt.imshow(mask)
    #plt.show()

    # disconnect certain components
    mask = si.binary_closing(mask)
    mask = si.binary_erosion(mask, structure=[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], iterations=10).astype(mask.dtype)
    mask = si.binary_dilation(mask, structure=[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], iterations=10).astype(mask.dtype)
    #plt.imshow(mask)
    #plt.show()


    mask = convex_hull_image(mask)

    #plt.imshow(img)
    #plt.imshow(mask, alpha=.2)
    #plt.show()

    return mask

def morphological_mask(img, cam, thresh=40, roi_1=(100, 280), roi_2=(100, 320)):
    W = img.copy()

    # compute gradient with non-maxima suppression
    smoothed = si.gaussian_filter(img, 1)
    gx, gy = np.gradient(img)
    gradient = np.hypot(gx, gy)
    #nonmax = si.convolve(gradient, NMS_FILTER(17), output = np.int64, mode = "reflect") <= 0
    #gradient *= (~nonmax)

    if cam == 1:
        W[:, :roi_1[0] - 1] = 0
        W[:, roi_1[1]:] = 0
    else:
        W[:, :roi_2[0] - 1] = 0
        W[:, roi_2[1]:] = 0

    mask = np.ones_like(W).astype(dtype="bool")
    if cam == 1:
        mask[110:150, roi_1[0]:roi_1[1]] = False
        edge_points(gradient, roi_1[0], 130, roi_1[1] - 1, 130)

        #cv.rectangle(W, (roi_1[0], 110), (roi_1[1], 150), (255, 255, 0))

    else:
        mask[110:150, roi_2[0]:roi_2[1]] = False
        edge_points(gradient, roi_2[0], 130, roi_2[1] - 1, 130)

        #cv.rectangle(W, (roi_2[0], 110), (roi_2[1], 150), (255, 255, 0))

    #plt.imshow(W)
    #plt.show()

    roi = np.ma.array(W, mask=mask)
    avg = roi.mean()
    W[W > avg + thresh] = 0
    W[W < avg - thresh] = 0
    W[W > 0] = 1
    #plt.imshow(W)
    #plt.show()

    smoothed = si.gaussian_filter(img, 1)
    gx, gy = np.gradient(smoothed)
    gradient = np.hypot(gx, gy)
    nonmax = si.convolve(gradient, NMS_FILTER(17), output = np.int64, mode = "reflect") <= 0
    gradient *= (~nonmax)
    W[gradient > 3] = 0
    #plt.imshow(W)
    #plt.show()

    #W = si.binary_erosion(W, structure=[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], iterations=1).astype(W.dtype)

    # remove components not connected to center of image
    width = W.shape[1]
    height = W.shape[0]
    start_x = round(width / 2)
    start_y = round(height / 2)
    # TODO: if not == 1 in center, take other pixel.
    blobs, labels = si.label(W, structure=np.array([[0, 1, 0],
                                                    [1, 1, 1],
                                                    [0, 1, 0]]))
    W[blobs != blobs[start_y, start_x]] = 0

    # horizontally grow mask back to original size
    #W = si.binary_dilation(W, structure=[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], iterations=1).astype(W.dtype)

    # take convex hull
    W = convex_hull_image(W)
    #plt.imshow(img)
    #plt.imshow(W, alpha=.2)
    #plt.show()
    return W   # note image unchanged, need to apply mask manually after feature extraction

def fingerfocus(img, roi, sigma = 1, hystd = (0,.1), min_area = 150, nms_order = 17):

    """ Background elimination method

    @return img (numpy.ndarray) : The resulting image
    """

    img = img.copy()

    debug = log.getEffectiveLevel() <= logging.DEBUG
    #debug = True
    xmin, xmax, ymin, ymax = roi

    if debug: show_uint16(img, "Original Image")

    cropped = img[xmin : xmax, ymin : ymax]

    if debug: show_uint16(cropped, "Cropped")

    smoothed = si.gaussian_filter(cropped, sigma)

    if debug: show_uint16(smoothed, "Smoothed")
    # print(np.max(smoothed), np.min(smoothed))

    # smoothed[smoothed > 150] = 0
    # smoothed[smoothed < 20] = 0
    # plt.imshow(smoothed)
    # plt.show()

    # Gradient

    gx, gy = np.gradient(smoothed)

    # print(np.max(gx), np.min(gx))
    # plt.imshow(np.abs(gx))
    # plt.show()

    gradient = np.hypot(gx, gy)
    gradient = (gradient / gradient.max() * 65535).astype(np.uint16)

    otsu = threshold_otsu(cropped, 1 << 16)
    μ_grad, σ_grad = gradient.mean(), gradient.std()

    if debug: show_uint16(gradient, "Gradient")

    #Non-maxima suppression

    nonmax = si.convolve(gradient, NMS_FILTER(nms_order), output = np.int64, mode = "reflect") <= 0
    gradient *= (~nonmax)

    if debug: show_bool(gradient, "Local Maxima")

    #Hysteresis

    if hystd is not None:

        hys_low = μ_grad - hystd[0] * σ_grad
        hys_high = μ_grad + hystd[1] * σ_grad
        log.info(f"Manual hysteresis thresholds : [{hys_low:.0f}, {hys_high:.0f}]")

    else:

        hys_low = .5 * otsu
        hys_high = otsu
        log.info(f"Otsu hysteresis thresholds : [{hys_low:.0f}, {hys_high:.0f}]")

    rubbish = gradient <= hys_low
    strong = gradient >= hys_high
    weak = (~rubbish) * (~strong)

    if debug: show_bool(strong, "Strong")
    if debug: show_bool(weak, "Weak")

    before = weak.sum()

    weak *= si.binary_dilation(strong, structure = np.array([[1, 1, 1],
                                                             [1, 0, 1],
                                                             [1, 1, 1]]))

    log.info(f"{before - weak.sum()}/{before} weak edge pixels filtered")

    gradient *= (~rubbish) * (strong + weak)
    if debug: show_bool(gradient, "Hysteresis")

    #Orientation Discontinuity suppression

    orientations = np.arctan2(gy, gx)
    laplace = si.laplace(smoothed, output = np.int64, mode = "reflect")
    laplace = (laplace / laplace.max() * 65535).astype(np.uint16)

    if debug: show_bool(orientations, "Orientations")
    if debug: show_uint16(laplace, "Laplacian")

    ogx, ogy = np.gradient(orientations)

    ogradient = np.hypot(ogx, ogy)
    ogradient = (ogradient / np.abs(ogradient).max() * np.pi)

    if debug: show_bool(ogradient / ogradient.max() * 2, "Orientations gradient")

    ofilters = ((ogradient < ogradient.mean())
             * (gradient > gradient.mean()))

    o2 = orientations.copy()
    o2 *= ofilters

    if debug: show_bool(o2, "Orientations filtered")

    gradient *= ofilters

    if debug: show_bool(gradient, "Orientations corrected gradient")

#    histo, bin_edges = np.histogram(orientations[ofilters], 360)
#
#    gradmax = np.zeros(gradient.shape)
#    coords = np.unravel_index(gradient.argmax(), gradient.shape)
#    for x in range(coords[0] - 5, coords[0] + 5):
#        for y in range(coords[1] - 5, coords[1] + 5):
#            gradmax[min(x, gradient.shape[0]), min(y, gradient.shape[1])] = 1
#    gradmax[np.unravel_index(gradient.argmax(), gradient.shape)]
#    if debug: show_bool(gradmax, "Gradient maximum")
#
#    most_common_orientation = bin_edges[np.argsort(histo)[-1]]
#    #orientations[np.unravel_index(gradient.argmax(), gradient.shape)]
#    #bin_edges[np.argsort(histo)[-1]]
#    #orientations[(gradient != 0) * (ogradient < ogradient.mean())].mean()
#    log.info(f"Most common orientation : {most_common_orientation}")
#    tol = 1e-2
#    is_most_common = (orientations > most_common_orientation - tol) * (orientations < most_common_orientation + tol)
#
#    filtered_orientations = orientations.copy()
#    filtered_orientations[~is_most_common] = 0
#    filtered_orientations[is_most_common] = 1
#
#    if debug: show_uint16(si.rotate(img, - most_common_orientation * 180 / np.pi), "Straightened")
#
#    log.info(f"Angle correction : {most_common_orientation * 180 / np.pi}")
#
#    show_bool(filtered_orientations, "Most common orientation")
#    plt.plot(bin_edges[:-1], histo)
#    plt.draw()
#    plt.pause(1)

    #Connected components
    blobs, labnbr = si.label(gradient, structure = np.array([[0, 1, 0],
                                                             [1, 1, 1],
                                                             [0, 1, 0]]))

    pixels = blobs.ravel()
    areas = np.bincount(pixels)[1:]
    kept_labels = np.argwhere(areas > min_area) + 1
    mask = np.isin(blobs, kept_labels)

    if debug:

        colors = np.zeros((*blobs.shape, 3), np.uint8)
        for i in range(1, labnbr):
            colors[blobs == i] = np.random.choice(range(256), size = 3)
        cv.imshow("Connected components", colors)

    if debug: show_bool(mask, f"Connected components > {min_area}")

    #Filling

    h, w = mask.shape

    mask[:h//2, :w//2] = si.binary_dilation(mask[:h//2, :w//2],
                                            structure = np.array([[0, 0, 0],
                                                                  [0, 1, 1],
                                                                  [0, 1, 0]]), iterations = -1)
    mask[:h//2, w//2:] = si.binary_dilation(mask[:h//2, w//2:],
                                            structure = np.array([[0, 0, 0],
                                                                  [1, 1, 0],
                                                                  [0, 1, 0]]), iterations = -1)
    mask[h//2:, :w//2] = si.binary_dilation(mask[h//2:, :w//2],
                                            structure = np.array([[0, 1, 0],
                                                                  [0, 1, 1],
                                                                  [0, 0, 0]]), iterations = -1)
    mask[h//2:, w//2:] = si.binary_dilation(mask[h//2:, w//2:],
                                            structure = np.array([[0, 1, 0],
                                                                  [1, 1, 0],
                                                                  [0, 0, 0]]), iterations = -1)

    if debug: show_bool(mask, f"Quadrant-bloated mask")

    mask = convex_hull_image(mask)

    if debug: show_bool(mask, "Final mask")

    img[:xmin] = img[xmax:] = 0
    img[:, :ymin] = img[:, ymax:] = 0
    img[xmin : xmax, ymin : ymax] *= mask

    ratio = 1 - mask.sum() / img.size

    if ratio > .6:
        log.warning(f"{ratio * 100 :.1f}% of the image was removed !")
    else:
        log.info(f"DONE : {ratio * 100 :.1f}% of the image was removed")

    if debug: show_uint16(img, "End result")

    # if debug: show_bool(extract_features(img), "Extracted")

    if debug:
        cv.waitKey()
        cv.destroyAllWindows()

    mask_full= np.zeros(img.shape, dtype = 'int')
    mask_full[xmin : xmax, ymin : ymax] = mask

    # plt.imshow(mask_full)
    # plt.show()
    # print(np.count_nonzero(mask_full), np.count_nonzero(mask_full==0), mask_full.size)
    #plt.imshow(img)
    #plt.imshow(mask_full, alpha=.2)
    #plt.show()
    return img,mask_full

def cannybration(img, roi = (73, 217, 10, 360)):

    xmin, xmax, ymin, ymax = roi

    cv.imshow("Original Image", img)

    cropped = img[xmin : xmax, ymin : ymax]

    cv.imshow("Cropped", cropped)

    t = threshold_otsu(cropped)
    low = t/2
    high = t

    def refresh(low, high):

        edges1 = canny(cropped, low_threshold = low, high_threshold = high)
        edges2 = cv.Canny(cropped.astype(np.uint8), low, high)

        hull1 = convex_hull_image(edges1) * cv.normalize(cropped, dst = None, alpha = 0, beta = 65535, norm_type = cv.NORM_MINMAX)
        hull2 = convex_hull_image(edges2) * cv.normalize(cropped, dst = None, alpha = 0, beta = 65535, norm_type = cv.NORM_MINMAX)

        trow = np.hstack((edges1, edges2))
        brow = np.hstack((hull1, hull2))

        cv.imshow("Cannybration", np.vstack((trow, brow)).astype(np.float64))

    def slider_low(v):

        refresh(v, high)

    def slider_high(v):

        refresh(low, v)

    cv.namedWindow("Cannybration")
    cv.createTrackbar("low", "Cannybration", 0, 255, slider_low)
    cv.createTrackbar("high", "Cannybration", 0, 255, slider_high)
    cv.waitKey()
    cv.destroyAllWindows()
