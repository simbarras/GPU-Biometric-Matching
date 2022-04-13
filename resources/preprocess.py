def histogram_equalization(image, mask):

    """
    Applies histogram equalization on the input image, returns filtered

    @param image (numpy.ndarray) : raw image to filter as 2D array of unsigned 8-bit integers
    @param mask (numpy.ndarray) : mask to normalize as 2D array of booleans

    @return (numpy.ndarray) : A 2D boolean array with the same shape and data type of
        the input image representing the filtered image.
    """

    from skimage.exposure import equalize_hist
    from skimage.exposure import rescale_intensity

    retval = rescale_intensity(equalize_hist(
        image, mask = mask), out_range = (0, 255))

    # make the parts outside the mask totally black
    retval[~mask] = 0

    return retval


