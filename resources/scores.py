import numpy as np
import scipy.signal as sp

def miurascore(model, probe, ch = 30, cw = 90, retmax = False):

    """ Computes the score between the probe and the model.

    @param model (numpy.ndarray): The model of the user to test the probe against
    @param probe (numpy.ndarray): The probe to test
    @param ch (int) : Maximum search displacement in y-direction.
    @param cw (int) : Maximum search displacement in x-direction.

    @return (float): Value between 0 and 0.5, larger value means a better match
    """

    I = probe.astype(bool)
    R = model.astype(bool)
    # plt.imshow(R)
    # plt.show()
    h, w = R.shape # same as I
    crop_R = R[ch:h - ch, cw:w - cw] # erode model by (ch, cw)
    # plt.imshow(crop_R)
    # plt.show()
    # correlates using scipy - fastest option available iff the self.ch and
    # self.cw are height (>30). In this case, the number of components
    # returned by the convolution is high and using an FFT-based method
    # yields best results. Otherwise, you may try  the other options bellow
    # -> check our test_correlation() method on the test units for more
    # details and benchmarks.
    N_c = sp.fftconvolve(I, np.rot90(crop_R, k = 2), 'valid')

    # plt.imshow(N_c)
    # plt.show()
    # 2nd best: use convolve2d or correlate2d directly;
    # Nm = sp.convolve2d(I, np.rot90(crop_R, k=2), 'valid')
    # 3rd best: use correlate2d
    # Nm = sp.correlate2d(I, crop_R, 'valid')

    # figures out where the maximum is on the resulting matrix
    t0, s0 = np.unravel_index(N_c.argmax(), N_c.shape)
    # print("maximum:", t0, s0)
    # normalizes the output by the number of pixels lit on the input
    # matrices, taking into consideration the surface that produced the
    # result (i.e., the eroded model and part of the probe)
    R_c = N_c[t0, s0] / (np.count_nonzero(crop_R)
                      + np.count_nonzero(I[t0:t0 + h - 2 * ch, s0:s0 + w - 2 * cw]))

    return (R_c, t0, s0) if retmax else R_c