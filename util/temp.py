import numpy as np
from util import find_nuclei


pred = np.load('/mnt/ccvl15/yixiao/histopathology/nuclei_8c/results/preds/M41_1_[10865.9204,27545.8886]_component_data_w_seg_1.npy')
pred_map, contours, pred_map_bp = find_nuclei(pred)
