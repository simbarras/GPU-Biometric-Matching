"""
" All the background elimination
" routines in one place
"""
import cv2 as cv
import logging
import numpy as np
import scipy.ndimage as si
from PIL import Image
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu
from skimage.feature import canny
from skimage.morphology import convex_hull_image
from .utils import show_bool, show_uint16, img_hist

log = logging.getLogger(__name__)

def edge_mask(img, cam, roi_1=(35, 355), roi_2=(55, 360)):
    """ Edge mask created by Simon, creates mask for each column of the image individually by using a scaled gradient
    threshold for each column. After that, performs some global morphological operations to get the final mask."""


    ################ HELPER FUNCTIONS ####################
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

        avg_1 = np.sum(img[:, x_1: x_1 + 1], axis=1)
        avg_1 = avg_1 / np.average(avg_1)
        a = (max_thresh(avg_1, f_1, "up", threshold))
        b = (max_thresh(avg_1, f_1, "down", threshold))
        return [(x_1, a), (x_1, b)]



    ###################### EDGE MASK ALGORITHM ####################
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

    mask = np.zeros_like(img)
    for ux, uy, dx, dy in zip(points_up_x, points_up_y, points_down_x, points_down_y):
        mask[uy : dy, ux] = 1

    # remove very bright spots
    mask[img > 240] = 0

    # disconnect certain components
    mask = si.binary_closing(mask)
    mask = si.binary_erosion(mask, structure=[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], iterations=10).astype(mask.dtype)
    mask = si.binary_dilation(mask, structure=[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], iterations=10).astype(mask.dtype)
    mask = convex_hull_image(mask)
    return mask

def fingerfocus(img, roi, sigma = 1, hystd = (0,.1), min_area = 150, nms_order = 17):
    """ Background elimination method (outdated)

    @return img (numpy.ndarray) : The resulting image
    """

    # nxn matrix of -1's with an n² - 1 at its center
    NMS_FILTER = lambda n: np.array([[-1] * n] * (n // 2)
                                    + [[-1] * (n // 2) + [n ** 2 - 1] + [-1] * (n // 2)]
                                    + [[-1] * n] * (n // 2))

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