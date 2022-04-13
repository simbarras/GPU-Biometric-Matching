from .extraction import *
def shift_to_CoM(image):
    """
    Applies shift to image: shift center of mass towards the physical center

    @param extracted image (numpy.ndarray of float64 0's and 1's)

    @return (numpy.ndarray) : Array with the same shape and data type as
        the input image representing the shifted image
    """

    img_h, img_w = image.shape
    CoM_img = si.measurements.center_of_mass(image)
    print("CoM original img: ",CoM_img)

    h_transl = img_h/2 - CoM_img[0]
    w_transl = img_w/2 - CoM_img[1]
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

