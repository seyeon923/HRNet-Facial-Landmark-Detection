# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# Modified by Kim Se-yeon(tpdussla93@gmail.com)
# ------------------------------------------------------------------------------

import cv2
import scipy.signal
import torch
import scipy
import scipy.misc
import numpy as np


MATCHED_PARTS = {
    "300W": ([1, 17], [2, 16], [3, 15], [4, 14], [5, 13], [6, 12], [7, 11], [8, 10],
             [18, 27], [19, 26], [20, 25], [21, 24], [22, 23],
             [32, 36], [33, 35],
             [37, 46], [38, 45], [39, 44], [40, 43], [41, 48], [42, 47],
             [49, 55], [50, 54], [51, 53], [62, 64], [61, 65], [68, 66], [59, 57], [60, 56]),
    "AFLW": ([1, 6],  [2, 5], [3, 4],
             [7, 12], [8, 11], [9, 10],
             [13, 15],
             [16, 18]),
    "COFW": ([1, 2], [5, 7], [3, 4], [6, 8], [9, 10], [11, 12], [13, 15], [17, 18], [14, 16], [19, 20], [23, 24]),
    "WFLW": ([0, 32],  [1,  31], [2,  30], [3,  29], [4,  28], [5, 27], [6, 26], [7, 25], [8, 24], [9, 23], [10, 22],
             [11, 21], [12, 20], [13, 19], [14, 18], [15, 17],  # check
             [33, 46], [34, 45], [35, 44], [36, 43], [37, 42], [
                 38, 50], [39, 49], [40, 48], [41, 47],  # elbrow
             [60, 72], [61, 71], [62, 70], [63, 69], [
                 64, 68], [65, 75], [66, 74], [67, 73],
             [55, 59], [56, 58],
             [76, 82], [77, 81], [78, 80], [87, 83], [86, 84],
             [88, 92], [89, 91], [95, 93], [96, 97])}


def fliplr_joints(x, width, dataset='aflw'):
    """
    flip coords
    """
    matched_parts = MATCHED_PARTS[dataset]
    # Flip horizontal
    x[:, 0] = width - x[:, 0]

    if dataset == 'WFLW':
        for pair in matched_parts:
            tmp = x[pair[0], :].copy()
            x[pair[0], :] = x[pair[1], :]
            x[pair[1], :] = tmp
    else:
        for pair in matched_parts:
            tmp = x[pair[0] - 1, :].copy()
            x[pair[0] - 1, :] = x[pair[1] - 1, :]
            x[pair[1] - 1, :] = tmp
    return x


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_transform(center: tuple[float, float], scale: float,
                  output_size: tuple[int, int], rot: float = 0):
    """
    General image processing functions

    - `center`: (x, y)
    - `output_size`: (w, h)
    - `rot`: angle in degree
    """
    # Generate transformation matrix
    h = 200 * scale
    target_cx = (output_size[0] - 1) / 2
    target_cy = (output_size[1] - 1) / 2

    t = np.zeros((3, 3))
    t[0, 0] = output_size[0] / h
    t[1, 1] = output_size[1] / h
    t[0, 2] = -output_size[0] / h * center[0] + target_cx
    t[1, 2] = -output_size[1] / h * center[1] + target_cy
    t[2, 2] = 1
    if rot != 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -target_cx
        t_mat[1, 2] = -target_cy
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform_pixel(pt: tuple[float, float] | np.ndarray, center: tuple[float, float],
                    scale: float, output_size: tuple[int, int],
                    invert: bool = False, rot: float = 0):
    """
    - `pt`: (x, y)
    - `center`: (x, y)
    - `output_size`: (w, h)
    - `rot`: angle in degree
    """
    # Transform pixel location to different reference
    t = get_transform(center, scale, output_size, rot=rot)
    if invert:
        t = np.linalg.inv(t)

    if not isinstance(pt, np.ndarray):
        pt = np.array(pt)

    if len(pt.shape) == 1:
        pt = pt[np.newaxis, :]

    new_pt = np.ones((len(pt), 3))
    new_pt[:, :2] = pt
    new_pt = np.dot(t, new_pt.T)
    return np.rint(np.squeeze(new_pt[:2, :])).astype(np.int64).T


def transform_preds(coords, center, scale, output_size):
    return torch.tensor(transform_pixel(
        coords, center, scale, output_size, invert=True))


def crop(img: np.ndarray, center: tuple[float, float], scale: float,
         output_size: tuple[int, int], rot: float = 0):
    """
    `center`: (x, y)
    `output_size`: (w, h)
    """
    cx, cy = center[0], center[1]

    # Preprocessing for efficient cropping
    h, w = img.shape[0], img.shape[1]
    sf = scale * 200.0 / output_size[1]
    if sf < 2:
        sf = 1
    else:
        new_size = int(np.floor(max(h, w) / sf))
        h = int(np.floor(h / sf))
        w = int(np.floor(w / sf))
        if new_size < 2:
            return torch.zeros(output_size[1], output_size[0], img.shape[2]) \
                if len(img.shape) > 2 else torch.zeros(output_size[1], output_size[0])
        else:
            img = cv2.resize(img, (w, h))  # (0-1)-->(0-255)
            cx = (cx + 1) / sf - 1
            cy = (cy + 1) / sf - 1
            scale = scale / sf

    # Upper left point
    ul = np.array(transform_pixel(
        [0, 0], (cx, cy), scale, output_size, invert=True))
    # Bottom right point
    br = np.array(transform_pixel(
        output_size, (cx, cy), scale, output_size, invert=True))

    # padding for in case of ul or br is out of range
    pad = max([0, -ul[0], -ul[1], -br[0], -br[1],
              ul[0] - w, ul[1] - h, br[0] - w, br[1] - h])

    if rot != 0:
        # Padding so that when rotated proper amount of context is included
        pad = max(pad, int(np.ceil(np.linalg.norm(
            br - ul) / 2 - float(br[1] - ul[1]) / 2)))

    padded_img = cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, value=0)

    if rot != 0:
        # Remove padding
        padded_img = cv2.warpAffine(padded_img,
                                    cv2.getRotationMatrix2D(
                                        ((ul[0] + br[0] + 2 * pad - 1) / 2,
                                         (ul[1] + br[1] + 2 * pad - 1) / 2), rot, 1),
                                    (padded_img.shape[1], padded_img.shape[0]))

    new_img = padded_img[ul[1] + pad: br[1] +
                         pad, ul[0] + pad: br[0] + pad]

    new_img = cv2.resize(new_img, output_size)
    return new_img


def generate_target(img, pt, sigma, label_type='gaussian'):
    x, y = pt
    x = round(x)
    y = round(y)

    # Generate gaussian
    tmp_size = int(np.ceil(sigma * 3))
    size = 2 * tmp_size + 1
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if label_type.upper() == 'GAUSSIAN':
        k = np.zeros((size, size))
        k[y0, x0] = 1
        k = cv2.GaussianBlur(k, (size, size), sigma)
        k /= k[y0, x0]
    else:
        xs = np.arange(0, size, 1, np.float32)
        ys = xs[:, np.newaxis]
        k = sigma / (((xs - x0) ** 2 + (ys - y0) ** 2 + sigma ** 2) ** 1.5)

    img[:] = 0
    padded_img = cv2.copyMakeBorder(
        img, y0, y0, x0, x0, cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, value=0)
    new_x = x + x0
    new_y = y + y0

    if new_x < 0 or new_x >= padded_img.shape[1] or new_y < 0 or new_y >= padded_img.shape[0]:
        return img

    padded_img[new_y, new_x] = 1

    img = scipy.signal.convolve2d(padded_img, k, "valid")

    return img
