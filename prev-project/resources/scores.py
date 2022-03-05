import numpy as np
import scipy.signal as sp

from .utils import shift

#R_c
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
    h, w = R.shape # same as I
    crop_R = R[ch:h - ch, cw:w - cw] # erode model by (ch, cw)

    # correlates using scipy - fastest option available iff the self.ch and
    # self.cw are height (>30). In this case, the number of components
    # returned by the convolution is high and using an FFT-based method
    # yields best results. Otherwise, you may try  the other options bellow
    # -> check our test_correlation() method on the test units for more
    # details and benchmarks.
    N_c = sp.fftconvolve(I, np.rot90(crop_R, k = 2), 'valid')
    # 2nd best: use convolve2d or correlate2d directly;
    # Nm = sp.convolve2d(I, np.rot90(crop_R, k=2), 'valid')
    # 3rd best: use correlate2d
    # Nm = sp.correlate2d(I, crop_R, 'valid')

    # figures out where the maximum is on the resulting matrix
    t0, s0 = np.unravel_index(N_c.argmax(), N_c.shape)

    # normalizes the output by the number of pixels lit on the input
    # matrices, taking into consideration the surface that produced the
    # result (i.e., the eroded model and part of the probe)
    R_c = N_c[t0, s0] / (np.count_nonzero(crop_R)
                      + np.count_nonzero(I[t0:t0 + h - 2 * ch, s0:s0 + w - 2 * cw]))

    return (R_c, t0, s0) if retmax else R_c

#R_m
def miurascore_true(model, probe, ch = 30, cw = 90, retmax = False):

    I = probe.astype(bool)
    R = model.astype(bool)
    h, w = R.shape # same as I
    crop_R = R[ch:h - ch, cw:w - cw]

    w_R = np.count_nonzero(crop_R)

    w_I = np.zeros((2*ch + 1, 2*cw + 1))
    for t in range(0, 2*ch + 1):
        for s in range(0, 2*cw + 1):
            w_I[t, s] = np.count_nonzero(I[t : t + h - 2*ch, s : s + w - 2*cw])

    IR = sp.fftconvolve(I, np.rot90(crop_R, k = 2), 'valid')

    N_m = w_I + w_R - 2 * IR

    t0, s0 = np.unravel_index(N_m.argmin(), N_m.shape)
    min_N_m = N_m[t0, s0]

    R_m = min_N_m / (np.count_nonzero(crop_R) + np.count_nonzero(I[t0:t0 + h - 2*ch, s0:s0 + w - 2*cw]))

    return ((1 - R_m)/2, t0, s0) if retmax else (1 - R_m)/2

#r_m
def miurascore_simple(model, probe, ch = 30, cw = 90, retmax = False):

    I = probe.astype(bool)
    R = model.astype(bool)
    h, w = R.shape # same as I
    crop_R = R[ch:h - ch, cw:w - cw]

    w_R = np.count_nonzero(crop_R)

    w_I = np.zeros((2*ch + 1, 2*cw + 1))
    for t in range(0, 2*ch + 1):
        for s in range(0, 2*cw + 1):
            w_I[t, s] = np.count_nonzero(I[t : t + h - 2*ch, s : s + w - 2*cw])

    IR = sp.fftconvolve(I, np.rot90(crop_R, k = 2), 'valid')

    N_m = (w_I + w_R - 2 * IR) / (2 * np.count_nonzero(crop_R))

    t0, s0 = np.unravel_index(N_m.argmin(), N_m.shape)
    r_m = N_m[t0, s0]

    return ((1 - r_m)/2, t0, s0) if retmax else (1 - r_m)/2

#R_m^*
def miurascore_exp(model, probe, ch = 30, cw = 90, retmax = False):

    I = probe.astype(bool)
    R = model.astype(bool)
    h, w = R.shape # same as I
    crop_R = R[ch:h - ch, cw:w - cw]

    w_R = np.count_nonzero(crop_R)

    w_I = np.zeros((2*ch + 1, 2*cw + 1))
    for t in range(0, 2*ch + 1):
        for s in range(0, 2*cw + 1):
            w_I[t, s] = np.count_nonzero(I[t : t + h - 2*ch, s : s + w - 2*cw])

    IR = sp.fftconvolve(I, np.rot90(crop_R, k = 2), 'valid')

    N_m = 1 - 2*IR / (w_I + w_R)

    t0, s0 = np.unravel_index(N_m.argmin(), N_m.shape)
    R_m = N_m[t0, s0]

    return ((1 - R_m)/2, t0, s0) if retmax else (1 - R_m)/2

#rho_m^*
def miurascore_z(model, probe, retmax = False, ch = 30, cw = 90):

    I = probe.astype(np.bool)
    R = model.astype(np.bool)
    h, w = R.shape # same as I

    #Nm = sp.convolve2d(I, np.rot90(crop_R, k=2), mode = 'full', boundary = "fill", fillvalue = 0)

    R = np.pad(R[ch:, cw:], ((0, ch), (0, cw)))

    Nm = np.zeros((2*ch + 1, 2*cw + 1))
    for t in range(0, 2*ch+1):
        for s in range(0, 2*cw+1):
            left = shift(I, t, s)
            Nm[t, s] = np.count_nonzero(left * R)

    t0, s0 = np.unravel_index(Nm.argmax(), Nm.shape)

    Nmm = Nm[t0, s0]

    Mr = 1 - 2 * Nmm / (R.sum() + I.sum())

    return ((1 - Mr)/2, t0, s0) if retmax else (1 - Mr)/2


def miurascore_L_n(model, probe, retmax = False, ch = 30, cw = 90, n = 2):

    I = probe.astype(bool)
    R = model.astype(bool)
    h, w = R.shape
    crop_R = R[ch:h - ch, cw:w - cw]

    N_c = sp.fftconvolve(I, np.rot90(crop_R, k = 2), 'valid')
    t0, s0 = np.unravel_index(N_c.argmax(), N_c.shape)
    R_c = (N_c**n).sum() / (np.count_nonzero(crop_R) + np.count_nonzero(I[t0:t0 + h - 2 * ch, s0:s0 + w - 2 * cw]))**n

    return (R_c**(1/n), t0, s0) if retmax else R_c**(1/n)


def miurascore_serge(model, probe, retmax = False, ch = 30, cw = 90):

    R = model.astype(np.float64)
    I = probe
    h, w = R.shape # same as I
    crop_R = R[ch:h - ch, cw:w - cw]

    Nm = sp.fftconvolve(I, np.rot90(crop_R, k = 2), 'valid')

    t0, s0 = np.unravel_index(Nm.argmax(), Nm.shape)

    Nmm = Nm[t0, s0]

    Mr = Nmm / (R.sum() + I.sum())

    return Mr, t0, s0 if retmax else Mr


def miurascore_eqserge(model, probe, retmax = False, ch = 30, cw = 90):

    I = probe.astype(bool)
    R = model.astype(bool)
    h, w = R.shape # same as I
    crop_R = R[ch:h - ch, cw:w - cw]

    min_Rm = crop_R.size
    for t in range(0, 2*ch+1):
        for s in range(0, 2*cw+1):

            Rm = np.count_nonzero(I[t : t + h - 2*ch, s : s + w - 2*cw] ^ crop_R)

            if Rm < min_Rm:
                min_Rm = Rm
                t0, s0 = t, s


    Mr = min_Rm / (np.count_nonzero(R) + np.count_nonzero(I))

    return ((1 - Mr)/2, t0, s0) if retmax else (1 - Mr)/2
