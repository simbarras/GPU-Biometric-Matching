#!/bin/python3
#
# @filename:  __init__.py
# @data:      01.09.2020
# @author:    Julien Corsin & Etienne Bonvin

# @brief  General-purpose helpers and other aliases. Mainly conversion functions and shorthands

import base64
import json
import logging
import numpy as np
import re
import requests
from typing import Dict, Any
from matplotlib import pyplot as plt
import cv2 as cv

def img_hist(img):
    plt.hist(img.ravel(), bins=256, range=[0, 255])
    plt.show()

def serialise(d: Dict[str, Any]) -> bytes:
    return json.dumps(d).encode("UTF-8")


def deserialise(s):
    return json.loads(s.decode("UTF-8"))


def bytes_to_b64str(b):
    return base64.b64encode(b).decode("UTF-8")


def b64str_to_bytes(s):
    return base64.b64decode(s.encode("UTF-8"))


def npimg_to_b64str(i):
    return bytes_to_b64str(i.tobytes())


def b64str_to_npimg(s, dt="uint8"):
    return np.frombuffer(b64str_to_bytes(s), dtype = dt)


def int_to_bytelist(i, nb = 2) :
    return list(i.to_bytes(nb, byteorder = "big"))


def uint_to_bytes(i):
    return i.to_bytes(max((i.bit_length() + 7) // 8, 1), byteorder = "big")


def int_to_bytes(i):
    return i.to_bytes((8 + (i + (i < 0)).bit_length()) // 8, byteorder = "big", signed = True)


def bytes_to_uint(b):
    return int.from_bytes(b, byteorder = "big")


def bytes_to_int(b):
    return int.from_bytes(b, byteorder = "big", signed = True)


def utf8str_to_bytelist(s):
    return list(s.encode("UTF-8"))


def bytelist_to_utf8str(bl):
    return bytes(bl).decode("UTF-8")


def utf8str_to_int(s):
    return bytes_to_int(s.encode("UTF-8"))


def scale_img(p, l, h):
    return cv.normalize(p, dst = None, alpha = l, beta = h, norm_type = cv.NORM_MINMAX)


def scale_uint16(p):
    return scale_img(p, 0, (1 << 16) - 1)


def scale_uint8(p):
    return scale_img(p, 0, (1 << 8) - 1)


def show_bool(img, name="Boolean image"):
    cv.imshow(name, img.astype(np.float64))


def show_uint16(img, name="Unsigned 16 bits image"):
    cv.imshow(name, scale_uint16(img))


def regex_count(rex, lst):
    return len([s for s in lst if re.compile(rex).match(s)])


def nanequal(a1, a2):
    return np.allclose(a1, a2, atol = 0, rtol = 0, equal_nan = True)


def quickstats(a):
    return f"({a.min():.2f}|{a.mean():.2f}|{a.std():.2f}|{a.max():.2f})"


# def shift(W, t, s):
#     return np.pad(W[t:, s:], ((0, t), (0, s)))

def wrap_around(W, t, s):
    center_x = round(W.shape[1] / 2)
    center_y = round(W.shape[0] / 2)
    t_x = center_x - s # x translation
    t_y = center_y - t # y translation

    return np.roll(W, (t_y, t_x), (0, 1))


def shift(W, t, s):
    if t >= 0 :
        if s >= 0 :
            return np.pad(W[t:, s:], ((0, t), (0, s)))
        else :
            return np.pad(W[t:, :s], ((0, t), (-s, 0)))
    else :
        if s>=0:
            return np.pad(W[:t, s:], ((-t, 0), (0,s)))
        else :
            return np.pad(W[:t, :s], ((-t, 0), (-s, 0)))

def digits(i, b):

    d = []

    while i >= b:

        i, r = divmod(i, b)
        d.append(r)

    d.append(i)

    return d[::-1] #Big-endian


def drawindow(b, ch, cw, t = 0, s = 0):

    """ Draws in the given binary image a rectangle of size (width - 2*cw, height - 2*ch)
        with upper left corner at (t,s)

        @param b (ndarray) : The binary image. Undefined behavior if the image is not binary to begin with

        @return b (ndarray) : The binary image with a rectangle drawn inside it
    """

    dtype = b.dtype
    (h, w) = b.shape
    b = b.astype(np.bool)

    b[(t , t + h - 2*ch), s : s + w - 2*cw] = True
    b[t : t + h - 2*ch, (s, s + w - 2*cw)] = True

    return b.astype(dtype)


def tile(a, nh, nw):

    h, w = a.shape
    sh = h / nh
    sw = w / nw

    if not (sh.is_integer() and sw.is_integer()):
        raise ValueError("Provided values do not divide the shape of the array")

    sh = int(sh)
    sw = int(sw)

    l = []
    for i in range(0, h, sh):
        for j in range(0, w, sw):
            l.append(a[i : i + sh, j : j + sw])

    return l


def query_json(host, query):

    """ Sends a JSON, receives a JSON.

    @param host (str) : The host to send the query to
    @param query (dict) : The json dictionary to send

    @return (dict) : The json response received, if no error was produced by the HTTP request
    @raise (LoggedError) : If the response is not 200 ok
    @raise (ConnectionError) : If the given host could not be reached
    """

    try:

        resp = requests.post(host,
                             headers = {"content-type" : "application/json"},
                             data = json.dumps(query))

        if resp.ok:
            return resp.json()
        else:
            raise LoggedError("Distant internal error", logging.getLogger(__name__))

    except requests.ConnectionError:

        raise ConnectionError(host + " could not be reached")


class LoggedError(Exception):

    """ Custom error class
        Automatically logs itself on instantiation
    """

    def __init__(self, message, logger):

        super().__init__(message)
        logger.error(message)


def run_length_encoding(raw):
    """
    Run length encoding algorithm on a string.

    :param raw: raw string to encode.
    :return: the encoded string.
    """

    if raw is None:
        return None
    if len(raw) == 0:
        return ""

    ref = raw[0]
    count = 1
    pointer = 1
    res = ""
    while pointer < len(raw):
        if raw[pointer] != ref:
            res = res + f"{ref}{count}."
            ref = raw[pointer]
            count = 1
        else:
            count += 1
        pointer += 1
    res = res + f"{ref}{count}."
    return res


def run_length_decoding(encoded):
    """
    Decodes a string encoded with the corresponding encoding algorithm.
    :param encoded: the encoded string.
    :return: the decoded string.
    """
    chunks = encoded.split(".")
    res = ""
    for chunk in chunks:
        if len(chunk) > 0:
            symbol = chunk[0]
            res = res + symbol * int(chunk[1:])
    return res


def binary_array_to_b64str(arr):
    """
    Turn a binary image into a base64 string and optimize length of final string.
    :param arr: binary image.
    :return: the result base64 string.
    """

    res = []
    for i in range(len(arr)):
        byte_row = []
        for j in range(0, len(arr[i]), 8):
            byte_row.append(int(arr[i][j] * 2**7 +
                            arr[i][j + 1] * 2**6 +
                            arr[i][j + 2] * 2**5 +
                            arr[i][j + 3] * 2**4 +
                            arr[i][j + 4] * 2**3 +
                            arr[i][j + 5] * 2**2 +
                            arr[i][j + 6] * 2**1 +
                            arr[i][j + 7]))
        res.append(byte_row)
    return bytes_to_b64str(bytes(bytearray(np.ravel(res).tolist())))


def b64str_to_binary_array(b64str, array_dimension):
    """
    Turn a b64 string into a binary array. The string must have been encoded using the corresponding method.
    :param b64str: the encoded string.
    :param array_dimension: dimension of the binary array.
    :return: the binary array.
    """

    def byte_to_bitarray(val):
        """
        Turn a byte into a binary array of eight elements encoded most significant bit first (MSB).
        :param val: the byte.
        :return: the corresponding binary array.
        """

        res = []
        curr = val
        for i in range(7, -1, -1):
            if curr >= 2**i:
                res.append(1)
                curr -= 2**i
            else:
                res.append(0)
        if not curr == 0:
            raise ValueError("Incorrect transformation from byte to bitarray")
        return res

    tmp = np.asarray(bytearray(b64str_to_bytes(b64str)))
    return np.ravel([byte_to_bitarray(val) for val in tmp]).reshape(array_dimension)


def binary_array_to_binary_str(arr):
    """
    Turn a
    """
    res = ""
    for row in arr:
        for value in row:
            res += f"{int(value)}"
    return res


def binary_str_to_binary_array(bstr, array_dimension):
    return np.reshape([int(char) for char in bstr], array_dimension)
