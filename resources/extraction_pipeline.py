from .utils import shift
from .background import *
from .scores import miurascore
from .extraction import *
from .preprocess import *
from .prealign import *
from .postprocess import *
from .postalign import *

log = logging.getLogger(__name__)

def extract_mask(img, cam, mask_method):
    if mask_method == "fingerfocus":
        img, mask = fingerfocus(img, roi=(40, 190, 10, 360)) # note image has changed (0 where not masked)
    elif mask_method == "morph":
        mask = morphological_mask(img, cam) # note image unchanged
    else:
        raise NotImplementedError()
    return img, mask

def prealign(data, mask, alignment_method):
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
    # TODO: add skeletonization method
    if process_method == "id":
        pass
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
    elif alignment_method == 'miura_matching':
        # compute the optimal params; comment out self-made functions in preprocess (biocore.py)
        # compute hamming distance between the two W, W_tilde_same
        ch = 30
        cw = 90
        score, t0, s0 = miurascore(model, img_features, retmax=True)
        img_features = shift(img_features, t0 - ch, s0 - cw)
    elif alignment_method == 'centre_of_mass':
        img_features = shift_to_CoM(img_features)
    else:
        raise NotImplementedError()
    return img_features

def run_pipeline(img_path, caching=0, cache_path="", mask_method="fingerfocus", prealign_method="id",
                 preprocess_method="hist_eq", extraction_method="maximum_curvature",
                 postprocess_method="id", postalign_method="id", model=None):
    """
    @param img_path: path to image where extraction is performed on
    @param caching: Denotes the position of caching:
        - 0 indicating no caching,
        - 1 indicating caching after mask extraction
        - 2 indicating caching after prealignment
        - 3 indicating caching after preprocessing
        - 4 indicating caching after feature extraction (often used)
        - 5 indicating caching after postprocessing
        - 6 indicating caching after the entire pipeline is executed.
    @param cache_path: Path to cache, preferably on local computer (to not clutter the git repo).
    @param model: only used for miura translation alignment
    @return: extracted and aligned feature vector.
    """

    # TODO caching: use different methods as folder structure, search for cached images and use longest match

    cam = int(img_path[-5])
    img = Image.open(img_path)
    img = np.asarray(img)

    img, mask = extract_mask(img, cam, mask_method)
    img, mask = prealign(img, mask, prealign_method)
    img, mask = preprocess(img, mask, preprocess_method)
    fv, mask = extract_features(img, mask, extraction_method)
    fv = postprocess(fv, postprocess_method)
    fv = postalign(fv, postalign_method, model)
    return fv

def run_parameterized_pipeline():
    # TODO: facilitate search of parameters.
    pass