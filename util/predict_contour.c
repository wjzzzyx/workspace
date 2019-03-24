#include <Python.h>
#include <stdint.h>
#include <malloc.h>
#include <stdio.h>

void predict_contour(int32_t *pred_map, int32_t *contour, int h, int w) {
    int32_t *contour_map = (int32_t *)calloc(h * w, sizeof(int32_t));

    for (int i = 0; i < h; ++i)
        for (int j = 0; j < w; ++j)
            if (contour[i * w + j] == 1) {
                //printf("%d %d\n", i, j);
                int l, r, u, d;
                l = j > 0 ? j - 1 : 0;
                r = j < w - 1 ? j + 1 : w - 1;
                u = i > 0 ? i - 1 : 0;
                d = i < h - 1 ? i + 1 : h - 1;
                if (pred_map[i * w + l] > 0)
                    contour_map[i * w + j] = pred_map[i * w + l];
                else if (pred_map[i * w + r] > 0)
                    contour_map[i * w + j] = pred_map[i * w + r];
                else if (pred_map[u * w + j] > 0)
                    contour_map[i * w + j] = pred_map[u * w + j];
                else if (pred_map[d * w + j] > 0)
                    contour_map[i * w + j] = pred_map[d * w + j];
                else if (pred_map[u * w + l] > 0)
                    contour_map[i * w + j] = pred_map[u * w + l];
                else if (pred_map[u * w + r] > 0)
                    contour_map[i * w + j] = pred_map[u * w + r];
                else if (pred_map[d * w + l] > 0)
                    contour_map[i * w + j] = pred_map[d * w + l];
                else if (pred_map[d * w + r] > 0)
                    contour_map[i * w + j] = pred_map[d * w + r];
                pred_map[i * w + j] = contour_map[i * w + j];
            }
    free(contour_map);
}

