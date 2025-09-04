import cv2
import numpy as np
import math
from skimage.morphology import disk

def compute_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def seg2bmap(seg):
    seg = seg.astype(np.uint8)
    h, w = seg.shape
    seg = np.pad(seg, ((1, 1), (1, 1)), mode='constant')
    bmap = np.zeros_like(seg)
    bmap[1:-1, 1:-1] = seg[1:-1, 1:-1]
    bmap = bmap != bmap[1:-1, 2:]  # horizontal
    bmap |= bmap != bmap[2:, 1:-1]  # vertical
    return bmap[1:-1, 1:-1].astype(np.uint8)

def compute_boundary_f(pred, gt, bound_pix):
    mask_b = seg2bmap(pred)
    gt_b = seg2bmap(gt)

    bound_disk = disk(bound_pix)
    mask_dil = cv2.dilate(mask_b, bound_disk)
    gt_dil = cv2.dilate(gt_b, bound_disk)

    gt_match = gt_b & mask_dil
    pred_match = mask_b & gt_dil

    n_pred = mask_b.sum()
    n_gt = gt_b.sum()

    if n_pred == 0 and n_gt > 0:
        return 0.0
    elif n_pred > 0 and n_gt == 0:
        return 0.0
    elif n_pred == 0 and n_gt == 0:
        return 1.0

    precision = pred_match.sum() / n_pred
    recall = gt_match.sum() / n_gt

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """
    
    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width > w | height > h | abs(ar1 - ar2) >
                0.01), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap