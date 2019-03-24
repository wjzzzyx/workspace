import numpy as np
cimport numpy as np

cdef extern from 'predict_contour.h':
    void predict_contour(int *pred_map, int *contour, int h, int w)

def _predict_contour(np.ndarray[np.int32_t, ndim=2, mode='c'] pred_map not None,
                    np.ndarray[np.int32_t, ndim=2, mode='c'] contour not None):
    h = pred_map.shape[0]
    w = pred_map.shape[1]
    predict_contour(<int *> np.PyArray_DATA(pred_map), <int *> np.PyArray_DATA(contour), h, w)
    # return np.array(pred_map)
    return pred_map