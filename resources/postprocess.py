from .extraction import *
from scipy import signal
from skimage.morphology import skeletonize
from .background import *

def closing(fv):
    fv = si.binary_closing(fv).astype("float")
    return fv

def skeletonize_fv(fv, min_area=10, dilation_iterations=3):
    # closing
    fv = si.binary_closing(fv)

    #plt.imshow(fv)
    #plt.show()

    # apply kernel
    kernel = np.outer(signal.windows.gaussian(3, 1), signal.windows.gaussian(3, 1))
    hor = signal.fftconvolve(fv, np.rot90(kernel, k = 2), 'same')
    #plt.imshow(hor)
    #plt.show()

    mx, mn = np.max(hor), np.min(hor)
    thresh = mx - (mx - mn) / 1.1
    hor[hor < thresh] = 0
    hor[hor > 0] = 1
    fv = hor.astype('uint8')
    #plt.imshow(fv)
    #plt.show()

    # skeletonize
    fv = skeletonize(fv)
    #plt.imshow(fv)
    #plt.show()

    # remove noise
    blobs, labnbr = si.label(fv, structure = np.array([[1, 1, 1],
                                                     [1, 1, 1],
                                                     [1, 1, 1]]))
    pixels = blobs.ravel()
    areas = np.bincount(pixels)[1:]
    kept_labels = np.argwhere(areas > min_area) + 1
    fv = np.isin(blobs, kept_labels)
    #plt.imshow(fv)
    #plt.show()

    if dilation_iterations > 0:
        fv = si.binary_dilation(fv, [[0, 1, 0], [1, 1, 1], [0, 1, 0]], iterations=dilation_iterations)
    fv = fv.astype(dtype="float")
    #plt.imshow(fv)
    #plt.show()

    return fv