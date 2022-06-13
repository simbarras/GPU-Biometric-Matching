from .background import *
from .scores import miurascore
from .preprocess import *
from .prealign import *
from .postprocess import *
from .postalign import *
from os.path import isfile
import os
from os.path import isdir
log = logging.getLogger(__name__)

def extract_mask(img, cam, mask_method):
    if mask_method == "fingerfocus":
        img, mask = fingerfocus(img, roi=(40, 190, 10, 360)) # note image has changed (0 where not masked)
    elif mask_method == "edge":
        mask = edge_mask(img, cam)
    else:
        raise NotImplementedError()
    return img, mask

def prealign(data, mask, alignment_method, cam=None):
    if alignment_method == "id":
        pass # identity alignment
    elif alignment_method == "leftmost_edge":
        data, mask = align_leftmost_edge(data,mask)
    elif alignment_method == "huang_normalization":
        data, mask = huang_normalization(data, mask, False, False)
    elif alignment_method == "huang_fingertip":
        data, mask = huang_normalization(data, mask, True, False)
    elif alignment_method == "huang_leftmost":
        data, mask = huang_normalization(data, mask, False, True)
    elif alignment_method == "translation":
        data, mask = translation_alignment(data, mask, cam)
    else:
        raise NotImplementedError()
    return data, mask

def preprocess(data, mask, process_method):
    """ Preprocesses an image given in the form of a numpy array
    The default preprocessing step is histogram equalization

    @param data (numpy.ndarray) : the numpy array to be preprocessed
    @param mask (numpy.ndarray of bool) : the mask representing the finger
    @return data (numpy.ndarray) : the preprocessed image
    @return mask (numpy.ndarray of bool) : the mask representing the finger
    """
    if process_method == "id":
        pass
    elif process_method == "hist_eq":
        data = histogram_equalization(data, mask)
    else:
        raise NotImplementedError()
    return data, mask

def extract_features(image, mask, extraction_method):
    """ Given an image, and a mask, extract the corresponding features.
    @param image (numpy.ndarray) : The images to extract veins from
    @return (numpy.ndarray of float64 0's and 1's) : The extracted image
    """
    if extraction_method == "maximum_curvature":
        image, mask = maximum_curvature(image, mask, sigma = 3)
    elif extraction_method == "repeated_line_tracking":
        image = repeated_line_tracking(image, mask)
    return image, mask

def postprocess(img_features, process_method):
    if process_method == "id":
        pass
    elif process_method == "skeletonize":
        img_features = skeletonize_fv(img_features)
    elif process_method == "skeletonize_thin":
        img_features = skeletonize_fv(img_features, dilation_iterations=0)
    elif process_method == "closing":
        img_features = closing(img_features)
    else:
        raise NotImplementedError()
    return img_features

def postalign(img_features, alignment_method, model=None):
    """
    @param img_features: extracted features img
    @param alignment_method: Alignment Method used on extracted image. Possible alignment methods:
        - miura_matching
        - centre_of_mass
        - ...
    @param model: only used for miura translation alignment
    @return: applies transformation to both model and probe. Note that miura matching does only transform probe.
    """
    if alignment_method == 'id':
        pass # identity alignment
    elif alignment_method == 'shift_to_mass':
        img_features = shift_to_M(img_features)
    elif alignment_method == 'miura_matching':
        if model is None: # processing model, doesn't need to be shifted
            return img_features

        # compute the optimal params; comment out self-made functions in preprocess (biocore.py)
        # compute hamming distance between the two W, W_tilde_same
        ch = 30
        cw = 90
        score, t0, s0 = miurascore(model, img_features, retmax=True)
        img_features = shift(img_features, t0 - ch, s0 - cw)
    elif alignment_method == 'center_of_mass':
        img_features = shift_to_CoM(img_features)
    elif alignment_method == "erode_com":
        img_features = shift_to_CoM(img_features, erode=True)
    else:
        raise NotImplementedError()
    return img_features


def extract_file_name(img_path):
    dir_list = img_path.split("/")
    return dir_list[-1][:-4]

def cache_seek(img_path, cache_path, mask_method, prealign_method,
                 preprocess_method, extraction_method, postprocess_method, postalign_method):
    directories = [cache_path + "/" + mask_method, prealign_method, preprocess_method, extraction_method,
                   postprocess_method, postalign_method]

    img = None,
    mask = None,
    level = len(directories) - 1
    for i in range(len(directories), 0, -1):
        path = directories[0]
        for j in range(1, i):
            path += "/" + directories[j]
        path += "/" + extract_file_name(img_path)
        if isfile(path + ".npy"):
            img = np.load(path + '.npy')
            if level <= 3: # only need mask up until feature extraction
                mask = np.load(path + '_mask.npy')
            break
        level = i - 1

    return img, mask, level

def cache_write(img, mask, level, img_path, cache_prefix):
    if not os.path.isdir(cache_prefix):
        os.system('mkdir ' + cache_prefix)

    # assumes cache folder hierarchy already existing
    file_name = extract_file_name(img_path)
    np.save(cache_prefix + file_name, img)
    if level <= 2:
        np.save(cache_prefix + file_name + "_mask", mask)


def run_pipeline(img_path, caching=False, cache_path="", mask_method="fingerfocus", prealign_method="id",
                 preprocess_method="hist_eq", extraction_method="maximum_curvature",
                 postprocess_method="id", postalign_method="id", model=None):
    """
    @param img_path: path to image where extraction is performed on
    @param caching: If true, every intermediary result is cached to corresponding cache_path (hierarchically)
    @param cache_path: Path to cache, preferably on local computer (to not clutter the git repo).
    @param model: only used for miura translation alignment
    @return: extracted and aligned feature vector.
    """


    # seek postaligned fv from cache. if not there, seek postprocessed etc.
    # returns mask and feature vector or only feature vector (and mask as None) and a level, specifying what to execute.

    cam = int(img_path[-5])
    level = 0
    img = None
    mask = None
    if caching:
        img, mask, level = cache_seek(img_path, cache_path, mask_method=mask_method, prealign_method=prealign_method,
                        preprocess_method=preprocess_method, extraction_method=extraction_method,
                        postprocess_method=postprocess_method, postalign_method=postalign_method)

    cache_prefix = cache_path + "/" + mask_method + "/"
    if level == 0:
        img = Image.open(img_path)
        img = np.asarray(img)
        img, mask = extract_mask(img, cam, mask_method)
        if caching:
            cache_write(img, mask, level, img_path, cache_prefix)
        level += 1

    cache_prefix = cache_prefix + prealign_method + "/"
    if level == 1:
        img, mask = prealign(img, mask, prealign_method, cam)
        if caching:
            cache_write(img, mask, level, img_path, cache_prefix)
        level += 1

    cache_prefix = cache_prefix + preprocess_method + "/"
    if level == 2:
        img, mask = preprocess(img, mask, preprocess_method)
        if caching:
            cache_write(img, mask, level, img_path, cache_prefix)
        level += 1

    cache_prefix = cache_prefix + extraction_method + "/"
    if level == 3:
        img, mask = extract_features(img, mask, extraction_method)
        if caching:
            cache_write(img, mask, level, img_path, cache_prefix)
        level += 1

    if level == 4:
        img = postprocess(img, postprocess_method)
        level += 1

    if level == 5:
        img = postalign(img, postalign_method, model)
    return img