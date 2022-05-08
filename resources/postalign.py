from .extraction import *



def shift_to_CoM(image, erode=False, local_min=True, roi=(0, 376), rec=False, depth=0, maxDepth=2):
    """
    Applies shift to image: shift center of mass towards the physical center

    @param extracted image (numpy.ndarray of float64 0's and 1's)

    @return (numpy.ndarray) : Array with the same shape and data type as
        the input image representing the shifted image
    """


    img_h, img_w = image.shape
    if erode:
        er_image = si.binary_closing(image.astype("bool"))
        er_image = si.binary_erosion(er_image, iterations=1)
        #er_image[:, 0:roi[0]] = 0
        #er_image[:, roi[1]:] = 0
        CoM_img = si.measurements.center_of_mass(er_image)
        #plt.imshow(er_image)
        #plt.show()
    else:
        CoM_img = si.measurements.center_of_mass(image)


    h_transl = img_h/2 - CoM_img[0]
    w_transl = img_w/2 - CoM_img[1]

    if local_min:
        a = np.sum(image, axis=1)
        a = si.gaussian_filter1d(a, sigma=4)

        start_x = CoM_img[0]
        x = round(start_x)
        while a[x - 1] < a[x] or a[x + 1] < a[x]:
            if a[x - 1] < a[x]:
                x = x - 1
            else:
                x = x + 1
        h_transl = img_h / 2 - x

        b = np.sum(image, axis=0)
        b = si.gaussian_filter1d(b, sigma=5)
        start_y = CoM_img[1]
        y = round(start_y)
        while b[y - 1] < b[y] or b[y + 1] < b[y]:
            if b[y - 1] < b[y]:
                y = y - 1
            else:
                y = y + 1
        plt.plot(a)
        plt.axvline(x=CoM_img[0])
        plt.axvline(x=x)
        w_transl = img_w / 2 - y
        plt.show()




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

    if rec and depth < maxDepth:
        a = np.sum(image, axis=1)
        plt.plot(a)
        plt.show()

        half_h = round(img_h / 2)
        half_w = round(img_w / 2)

        img1 = translated_img[:half_h, :half_w]
        img1 = shift_to_CoM(img1, erode=erode, depth=depth+1, rec=True)
        img2 = translated_img[half_h:, :half_w]
        img2 = shift_to_CoM(img2, erode=erode, depth=depth+1, rec=True)
        img3 = translated_img[:half_h, half_w:]
        img3 = shift_to_CoM(img3, erode=erode,depth=depth+1, rec=True)
        img4 = translated_img[half_h:, half_w:]
        img4 = shift_to_CoM(img4, erode=erode,depth=depth+1, rec=True)
        print(img3.shape)
        translated_img[:half_h, :half_w] = img1
        translated_img[half_h:, :half_w] = img2
        translated_img[:half_h, half_w:] = img3
        translated_img[half_h:, half_w:] = img4

    return translated_img

