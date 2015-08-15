# --------------------------------------------------------
# The original code is reviced, so that the drawbacks of 
# the datasets could be overcome
#   Fast R-CNN
#   Copyright (c) 2015 Microsoft
#   Licensed under The MIT License [see LICENSE for details]
#   Written by Sergey Karayev
# --------------------------------------------------------

cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def bbox_overlaps(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes,
        int type_class=-1, float percent=0.5):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, n
    print "We are in the function!!!!!!!!!!"
    if type_class == -1 :
        # the default one, ignore the type of class
        for k in range(K):
            box_area = (
                (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                (query_boxes[k, 3] - query_boxes[k, 1] + 1)
            )
            for n in range(N):
                iw = (
                    min(boxes[n, 2], query_boxes[k, 2]) -
                    max(boxes[n, 0], query_boxes[k, 0]) + 1
                )
                if iw > 0:
                    ih = (
                        min(boxes[n, 3], query_boxes[k, 3]) -
                        max(boxes[n, 1], query_boxes[k, 1]) + 1
                    )
                    if ih > 0:
                        ua = float(
                            (boxes[n, 2] - boxes[n, 0] + 1) *
                            (boxes[n, 3] - boxes[n, 1] + 1) +
                            box_area - iw * ih
                        )
                        overlaps[n, k] = iw * ih / ua
    else:
        # in this case, we consider the class type, 1 for upper, 2 for lower
        for k in range(K):  # k is the gt box number
            box_area = (
                (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                (query_boxes[k, 3] - query_boxes[k, 1] + 1)
            )
            for n in range(N):
                iw = (
                    min(boxes[n, 2], query_boxes[k, 2]) -
                    max(boxes[n, 0], query_boxes[k, 0]) + 1
                )
                if iw > 0:
                    ih = (
                        min(boxes[n, 3], query_boxes[k, 3]) -
                        max(boxes[n, 1], query_boxes[k, 1]) + 1
                    )
                    if ih > 0:
                        ua = float(
                            (boxes[n, 2] - boxes[n, 0] + 1) *
                            (boxes[n, 3] - boxes[n, 1] + 1) +
                            box_area - iw * ih
                        )
                        overlaps[n, k] = iw * ih / ua
                        if type_class == 1:
                            # the possible lower part
                            if percent * boxes[n, 3] + (1 - percent) * boxes[k, 1] > 2 * query_boxes[k, 3]:
                                if  2 * query_boxes[k, 0] < boxes[n, 0] + boxes[n, 2] < 2 * query_boxes[k, 2]:
                                    overlaps[n, k] = 0
                        else:
                            if type_class == 2:
                                if percent * boxes[n, 3] + (1 - percent) * boxes[k, 1] > 2 * query_boxes[k, 3]:
                                    if  2 * query_boxes[k, 0] < boxes[n, 0] + boxes[n, 2] < 2 * query_boxes[k, 2]:
                                        overlaps[n, k] = 0
                                # the possible upper part
    return overlaps

def bbox_coverage(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, n

    for k in range(K):
        for n in range(N):
            if query_boxes[k, 0] >= 0:  # if the coordinate doesnt exist, it's set to -1
                iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                    max(boxes[n, 0], query_boxes[k, 0]) + 1)
                if iw > 0:
                    ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                        max(boxes[n, 1], query_boxes[k, 1]) + 1)
                    if ih > 0:
                        volume = iw * ih
                        size1 = (
                            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
                            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
                        )
                        size2 = (
                            (boxes[k, 2] - boxes[k, 0] + 1) *
                            (boxes[k, 3] - boxes[k, 1] + 1)
                        )
                        size = min(size1, size2)
                        overlaps[n, k] = iw * ih / size
    return overlaps
