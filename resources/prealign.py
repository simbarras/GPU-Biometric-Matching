from .extraction import *
from .utils import *
import numpy as np

def align_leftmost_edge(image, mask):
    constant_x = 45

    img_h, img_w = image.shape

    edges = np.zeros((2, mask.shape[1]), dtype=int)

    edges[0,:] = mask.argmax(axis=0) # get upper edges
    edges[1,:] = len(mask) - np.flipud(mask).argmax(axis=0) - 1

    for i in range(0,edges.shape[1]):
        if(edges[1][i] - edges[0][i] <= img_h/4 and edges[0][i]<img_h/2 and edges[1][i]>img_h/2):
            break


    x_transl = constant_x - i if abs(constant_x - i) < 50 else 0
    y_transl = 0

    def _transl(img,x,y):
        '''Applies the translation with h_transl, w_transl on the input image'''
        translation_matrix = np.float32([
        [1, 0, x],
        [0, 1, y]])
        shifted = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
        return shifted #np.array(shifted).astype(img.dtype)

    return _transl(image,x_transl,y_transl), _transl(mask.astype(np.float32),x_transl,y_transl).astype(np.int64)

def huang_normalization(image, mask, fingertip, leftedge):
    '''Simple finger normalization from Huang et. al

    Based on B. Huang, Y. Dai, R. Li, D. Tang and W. Li, Finger-vein
    authentication based on wide line detector and pattern normalization,
    Proceedings on 20th International Conference on Pattern Recognition (ICPR),
    2010.

    This implementation aligns the finger to the centre of the image using an
    affine transformation. Elliptic projection which is described in the
    referenced paper is **not** included.

    In order to defined the affine transformation to be performed, the
    algorithm first calculates the center for each edge (column wise) and
    calculates the best linear fit parameters for a straight line passing
    through those points.

    '''
    padding_width = 5
    padding_constant = 51
    fingertip_constant = 350
    leftedge_constant = 45
    img_h, img_w = image.shape

    # Calculates the mask edges along the columns
    edges = np.zeros((2, mask.shape[1]), dtype=int)

    edges[0,:] = mask.argmax(axis=0) # get upper edges
    edges[1,:] = len(mask) - np.flipud(mask).argmax(axis=0) - 1

    x_transl = 0

    if (fingertip):
        for i in range(edges.shape[1]-1,0,-1):
            if(edges[1][i] - edges[0][i] <= img_h/4 and edges[0][i]<img_h/2 and edges[1][i]>img_h/2):
                break

        x_transl = fingertip_constant - i if abs(fingertip_constant - i) < 50 else 0

    if (leftedge):
        for i in range(0,edges.shape[1]):
            if(edges[1][i] - edges[0][i] <= img_h/4 and edges[0][i]<img_h/2 and edges[1][i]>img_h/2):
                break

        x_transl = leftedge_constant - i if abs(leftedge_constant - i) < 50 else 0

    bl = edges.mean(axis=0) #baseline
    x = np.arange(0, edges.shape[1])
    A = np.vstack([x, np.ones(len(x))]).T

    # Fit a straight line through the base line points
    w = np.linalg.lstsq(A,bl)[0] # obtaining the parameters
    # plt.plot(x, bl, 'o', label='Original data', markersize=10)
    # plt.plot(x, w[0]*x + w[1], 'r', label='Fitted line')
    # plt.show()
    angle = -1*math.atan(w[0])  # Rotation
    tr = img_h/2 - w[1]         # Translation
    scale = 1.0                 # Scale

    #Affine transformation parameters
    sx=sy=scale
    cosine = math.cos(angle)
    sine = math.sin(angle)

    a = cosine/sx
    b = -sine/sy
    #b = sine/sx
    c = x_transl

    d = sine/sx
    e = cosine/sy
    f = tr #Translation in y
    #d = -sine/sy
    #e = cosine/sy
    #f = 0

    g = 0
    h = 0
    #h=tr
    i = 1

    T = np.matrix([[a,b,c],[d,e,f],[g,h,i]])
    # print(T)
    Tinv = np.linalg.inv(T)
    Tinvtuple = (Tinv[0,0],Tinv[0,1], Tinv[0,2], Tinv[1,0],Tinv[1,1],Tinv[1,2])

    def _afftrans(img):
      '''Applies the affine transform on the resulting image'''

      t = Image.fromarray(img.astype('uint8'))
      w, h = t.size #pillow image is encoded w, h
      w += 2*padding_width
      h += 2*padding_width
      t = t.transform(
          (w,h),
          Image.AFFINE,
          Tinvtuple,
          resample=Image.BICUBIC,
          fill=padding_constant)

      return np.array(t).astype(img.dtype)
    image = _afftrans(image)
    plt.imshow(image)
    plt.show()
    return image, _afftrans(mask)


def Rotate(rotateImage, angle, x, y):
    # Taking image height and width
    imgHeight, imgWidth = rotateImage.shape[0], rotateImage.shape[1]

    # Computing 2D rotation Matrix to rotate an image
    rotationMatrix = cv2.getRotationMatrix2D((y, x), angle, 1.0)

    # Now, we will perform actual image rotation
    rotatingimage = cv2.warpAffine(
        rotateImage, rotationMatrix, (imgWidth, imgHeight))

    return rotatingimage


def get_y_vec(img, axis=0):
    n = img.shape[axis]
    s = [1] * img.ndim
    s[axis] = -1
    i = np.arange(n).reshape(s)
    return np.round(np.sum(img * i, axis=axis) / np.sum(img, axis=axis), 1)

def cols_to_com(mask, img):
    # calculate CoM for each column of mask. make sure it is center of image if no mask pixels available.
    pass


def translation_alignment(image, mask, cam, roi_1=(100, 300), roi_2=(100, 300)):
    #plt.imshow(image)
    #plt.imshow(mask, alpha=.2)
    #plt.show()

    mask = mask.astype(dtype="float")

    # find out where principal axis of the finger is, translate to center of image.
    from sklearn.linear_model import LinearRegression
    Y, X = np.where(mask == 1)
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X.reshape(-1, 1), Y.reshape(-1, 1))
    lr_yhat = X * lr.coef_[0] + lr.intercept_
    #plt.imshow(image)
    #plt.plot(range(np.min(X), np.max(X)), get_y_vec(mask)[np.min(X):np.max(X)])
    #plt.plot(X, lr_yhat, 'r-', label='fit_intercept=False')
    #plt.scatter([np.average(X)], [np.average(Y)])
    #plt.show()
    centerY, centerX = image.shape[0] // 2, image.shape[1] // 2

    line_centerX, line_centerY = np.average(X), np.average(Y)

    x_s, y_s = int(centerX - line_centerX), int(centerY - line_centerY)

    angle = 360 * math.atan(lr.coef_[0]) / (2 * math.pi)
    image = Rotate(image, angle, line_centerX, line_centerY)
    mask = Rotate(mask, angle, line_centerX, line_centerY)
    image = shift(image, -y_s, -x_s)
    mask = shift(mask, -y_s, -x_s)

    plt.imshow(image)
    plt.show()

    return image, mask.astype(dtype="uint16")