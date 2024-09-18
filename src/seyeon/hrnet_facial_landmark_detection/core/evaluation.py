# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import math

import torch
import numpy as np

from ..utils.transforms import transform_preds


def get_preds(scores):
    """
    get predictions from score maps in torch Tensor
    return type: torch.LongTensor
    """
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1)

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0]) % scores.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1]) / scores.size(3))

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask
    return preds


def compute_nme(preds, meta):

    targets = meta['pts']
    preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(
            pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse


def decode_preds(output, center, scale, res):
    preds = get_preds(output)  # float type

    preds = preds.cpu()
    # pose-processing
    for n in range(preds.size(0)):
        for p in range(preds.size(1)):
            hm = output[n, p, :]
            assert hm.shape == (res[1], res[0])

            px = round(preds[n, p, 0].item())
            py = round(preds[n, p, 1].item())
            if (px > 0) and (px < res[0] - 1) and (py > 0) and (py < res[1] - 1):
                diff = torch.Tensor(
                    [hm[py][px + 1] - hm[py][px - 1], hm[py + 1][px] - hm[py - 1][px]])
                preds[n, p] += diff.sign() * .25

    # Transform back
    for i in range(preds.size(0)):
        preds[i, :] = transform_preds(preds[i], center[i], scale[i], res)

    return preds
