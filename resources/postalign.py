import numpy as np
import scipy.signal as sp
from .extraction import *
from .utils import shift
from .utils import wrap_around

quadrant = None

def set_quadrant(i):
    global quadrant
    quadrant = i

def find_mass_point(N_c, u_l_x, u_l_y):
    t0, s0 = np.unravel_index(N_c.argmax(), N_c.shape)
    return u_l_x + s0, u_l_y + t0

def get_quadrant(quadrant, img):
    q = quadrant
    if type(quadrant) == list:
        quadrant = quadrant.copy()
        q = quadrant.pop()

    half_x = round(img.shape[1] / 2)
    half_y = round(img.shape[0] / 2)
    x, y = 0, 0
    x_end, y_end = half_x, half_y
    if(q >= 2):
        y = half_y
        y_end = img.shape[0]
    if(q % 2 == 1):
        x = half_x
        x_end = img.shape[1]

    ret_img = img[y : y_end, x : x_end]

    if type(quadrant) == list and len(quadrant) > 0:
        return get_quadrant(quadrant, ret_img)

    return ret_img, x, y

def shift_to_M(img, k_y = 5, k_x = 10, q=None, kernel=None):
    global quadrant
    if q is not None:
        quadrant = q

    if kernel is None:
        kernel = np.ones((k_y, k_x))
    N_c = sp.fftconvolve(img, np.rot90(kernel, k=2), 'valid')

    if quadrant is not None:
        x, y = find_mass_point(*get_quadrant(quadrant, N_c))
    else:
        y, x = np.unravel_index(N_c.argmax(), N_c.shape)

    img_features = wrap_around(img.astype("uint16"), y, x)
    # from .postprocess import skeletonize_fv
    # img_features = skeletonize_fv(img_features)
    return img_features


def shift_to_CoM(image):
    """
    Applies shift to image: shift center of mass towards the physical center

    @param extracted image (numpy.ndarray of float64 0's and 1's)

    @return (numpy.ndarray) : Array with the same shape and data type as
        the input image representing the shifted image
    """

    img_h, img_w = image.shape
    CoM_img = si.measurements.center_of_mass(image)


    h_transl = img_h/2 - CoM_img[0]
    w_transl = img_w/2 - CoM_img[1]
    print("CoM original img: ",CoM_img)

    # print("Original img center: ",img_h/2, img_w/2)
    # print("To translate by: ",h_transl, w_transl, round(h_transl),round(w_transl))

    def _transl(img,x,y):
      '''Applies the translation with h_transl, w_transl on the input image'''
      translation_matrix = np.float32([
	     [1, 0, x],
	     [0, 1, y]])

      shifted = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
      return shifted #np.array(shifted).astype(img.dtype)

    translated_img = _transl(image,round(w_transl),round(h_transl))
    # print("Translated img center ",translated_img.shape[0]/2,translated_img.shape[1]/2)
    CoM_timg = si.measurements.center_of_mass(translated_img)
    # print("CoM translated img: ",CoM_timg)

    return translated_img

