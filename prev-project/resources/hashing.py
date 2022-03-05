import cv2 as cv
import logging
import numpy as np
import os
import pdb
# import sage
import scipy.linalg as la
import scipy.ndimage as si
import scipy.signal as sp
import sys
# import tqdm
from time import time

from utils import show_bool, shift
# from biometrics.scores import miurascore
from scores import miurascore
from background import *
from PIL import Image

log = logging.getLogger()
log.setLevel(logging.INFO)

logstream = logging.StreamHandler(sys.stdout)
# logfile = logging.FileHandler(f"hashing.txt", "w+")

log.addHandler(logstream)
# log.addHandler(logfile)
import time


def rand_M(h, w, ch, cw):
    rng = np.random.default_rng()
    M = rng.integers(low=-1, high=1, size=(h, w))
    M = 2 * M + 1
    M[:ch, :] = 0
    M[h - ch:, :] = 0
    M[:, :cw] = 0
    M[:, w - cw:] = 0
    return M


def generate_K(d, h=240, w=376, ch=30, cw=90, K_folder="../data/Ks/"):
    """
    Generates and stores batches of 500 random matrices M of size 240*376
    as described in the hashing algorithm.
    parameters :
     - d : hash size, must be a multiple of 500
     - h x w : image size : 240 x 376
     - ch, cw : convolution window params, ch = 30, cw = 90
     - K_folder : folder where the key will be stored
    """
    assert d < 500 or d % 500 == 0, "d should be a multiple of 500"
    n_batch = max(d // 500, 1)
    for i in range(n_batch):
        # Keys should only be generated once ! Else all hashes have to be recompute
        if os.path.isfile(K_folder + "K_{}.npy".format(i)):
            print("Key batch {} was already generated.".format(i))
        else:
            K = np.array([rand_M(h, w, ch, cw) for _ in range(500)])
            np.save(K_folder + "K_{}.npy".format(i), K)
            print("New key batch generated and stored !")


def Hash(W, Ks, ch, cw, d, eps):
    h, w = W.shape

    if d <= 500:
        K = Ks['K_0'][:d]
        kw = np.einsum("kij,ij->k", K[:, ch:h - ch, cw:w - cw], W[ch:h - ch, cw:w - cw])
        if eps > 0: kw[np.abs(kw) <= eps] = 0
        return np.sign(kw)
    else:
        hkw = np.array([])
        n_batch = d // 500
        for i in range(n_batch):
            kw = np.einsum("kij,ij->k", Ks['K_{}'.format(i)][:, ch:h - ch, cw:w - cw], W[ch:h - ch, cw:w - cw])
            if eps > 0: kw[np.abs(kw) <= eps] = 0
            hkw = np.concatenate((hkw, np.sign(kw)))
        return hkw


def shifting_factors(H, W_tilde, t_max=15, s_max=30, ch=30, cw=90, d=100, eps=0):
    H = H[:d]

    max_V = 0

    t_opt = 0
    s_opt = 0

    Ks = {'K_{}'.format(i): np.load("../data/Ks/K_{}.npy".format(i)) for i in range(max(1, d // 500))}

    # for t in tqdm.tqdm(range(-t_max, t_max)):
    for t in range(-t_max, t_max):
        for s in range(-s_max, s_max):
            H_tilde = Hash(shift(W_tilde, t, s), Ks, ch, cw, d, eps)

            if (V := H_tilde @ H) > max_V:
                max_V = V
                t_opt = t
                s_opt = s

    return (t_opt, s_opt)


def V_K(H, H_tilde, d):
    assert len(H) == d
    assert len(H_tilde) == d
    return H_tilde @ H


if __name__ == "__main__":
    """
    Hashing algorithm :
    1) choose d and ε
    2) Key generation and storage :
        generate_K(d = d, K_folder = "../data/Ks/", h = 240, w = 376...)

    Enroll
    3) Load Reference sample W and do preprocessing :
        - W = np.load(image_folder + image_name)
        - W = background_elimination(W)
        - W = extract_features(W)
    4) Compute hash of W :
        - H = H(K, W, ch, cw, ε)

    Capture
    5) repeat 3) for new sample W_tilde
    6) compute optimal shifting factors :
        s, t = shifting_factors(H, W_tilde, t_max, s_max, ch, cw, d, eps)
        -- using s,t from miurascore instead (faster) and sufficient for my purpose --
    7) compute similarity score :
        V = V_K(H, W_tilde, t, s, ch, cw, d, eps)
        
    """
    ratios = []
# middle, ring, index, (little, thumb)
# cam1, cam2
    S_folder="../data/Ss"#forgot a slash oops;now they get stored inside data with prefix
    H_folder="../data/Hs"
    for user in ['5','3']:
        for finger in ['middle','ring', 'index']:
            for cam in ['cam1','cam2']:
                for lr in ['left','right']:
                    img_trial1 = "../../archive/" + user + "_" + lr + "_" + finger + "_1_" + cam + ".png"
                    img_trial2 = "../../archive/" + user + "_" + lr + "_"  + finger + "_2_" + cam + ".png"
                    if img_trial1 != "../../archive/5_left_middle_1_cam1.png" and img_trial1 != "../../archive/5_left_middle_1_cam2.png" and img_trial1 != "../../archive/5_left_index_1_cam1.png" and img_trial1 != "../../archive/5_left_index_1_cam2.png":
                        print("Load and extract W", img_trial1)
                        W = Image.open(img_trial1)
                        W = np.asarray(W)
                        W, mask = fingerfocus(W, roi=(40, 190, 10, 360))
                        W, mask = extract_features(W, mask)
                        # np.save("../../archive/extracted/5_left_ring_1_cam1.png", W)

                        print("Load and extract W_tilde_same", img_trial2)
                        W_tilde_same = Image.open(img_trial2)
                        W_tilde_same = np.asarray(W_tilde_same)
                        W_tilde_same, mask_tilde = fingerfocus(W_tilde_same, roi = (40, 190, 10, 360))
                        W_tilde_same, mask_tilde = extract_features(W_tilde_same, mask_tilde)
                        # np.save("../../archive/extracted/5_left_ring_2_cam1.png", W_tilde_same)

                        h, w = W.shape
                        ch, cw = 30, 90

                        eps = 0# 23
                        d = 3500

                        print("Key generation")
                        generate_K(d = d, h = 240, w = 376, ch = 30, cw = 90, K_folder="../data/Ks/")

                        print("Loading the generated keys")
                        Ks = {'K_{}'.format(i): np.load("../data/Ks/K_{}.npy".format(i)) for i in range(max(1, d // 500))}

                        # print("Hashing of W for shifting")
                        # H = Hash(W, Ks=Ks, ch=ch, cw=cw, d=1000, eps=eps)
                        #
                        # print("computing the shifting factors using biometric hashing")
                        # t, s = shifting_factors(H, W_tilde_same, t_max = 15, s_max = 30, d = 1000, eps = eps) #eps = 0
                        # print(t,s)
                        score, t0, s0 = miurascore(W, W_tilde_same, retmax=True)

                        print("Hashing of W")
                        H = Hash(W, Ks=Ks, ch=ch, cw=cw, d=d, eps=eps)
                        np.save(H_folder + "H_{}_{}_{}_{}_{}.npy".format(user,finger,cam,lr,'1'), H)

                        print(t0-ch,s0-cw)
                        print("Hashing of W_tilde_same")
                        # H_tilde_same = Hash(shift(W_tilde_same, t, s), Ks=Ks, ch=ch, cw=cw, d=d, eps=eps)
                        H_tilde_same = Hash(shift(W_tilde_same,t0-ch,s0-cw), Ks=Ks, ch=ch, cw=cw, d=d, eps=eps)
                        np.save(H_folder + "H_{}_{}_{}_{}_{}.npy".format(user,finger,cam,lr,'2'), H_tilde_same)

                        V = V_K(H, H_tilde_same, d)
                        # print(V)

                        div = H*H_tilde_same # elmt div -1*-1 gives 1 same as 1*1; rest gives -1

                        print(np.unique(div), np.unique(H), np.unique(H_tilde_same))


                        nr_of_ones = np.count_nonzero(div == 1)
                        print("nr of 1s in prod: ",nr_of_ones)
                        anything_but_1 = d - nr_of_ones
                        different_pixels = anything_but_1/d
                        print("ratio of difrt: ", different_pixels)
                        ratios.append(different_pixels)
    mean_difrt_pix = np.mean(np.asarray(ratios))
    print(mean_difrt_pix)
    print(ratios)

    # if V < (smth that creates) EER : print("Authentication failed)
    # if V >= EER : print("Authentication succeeded)
