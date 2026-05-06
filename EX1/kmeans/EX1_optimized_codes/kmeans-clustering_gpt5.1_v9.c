#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "../kmeans.h"

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

__inline float euclid_dist_2(float *pt1, float *pt2, int numdims) {
    int i;
    float ans = 0.0f;

    for (i = 0; i < numdims; i++) {
        float diff = pt1[i] - pt2[i];
        ans += diff * diff;
    }

    return ans;
}

int find_nearest_point(float *pt, int nfeatures, float **pts, int npts) {
    int index = 0;
    int i;
    float min_dist = FLT_MAX;

    for (i = 0; i < npts; i++) {
        float dist = euclid_dist_2(pt, pts[i], nfeatures);
        if (dist < min_dist) {
            min_dist = dist;
            index = i;
        }
    }
    return index;
}

float **kmeans_clustering(float **feature,
                          int nfeatures, int npoints, int nclusters,
                          float threshold, int *membership)
{
    int i, j, n = 0, index, loop = 0;
    int *new_centers_len;
    float **new_centers;
    float **clusters;
    float delta;

    clusters = (float **)malloc((size_t)nclusters * sizeof(float *));
    if (!clusters)
        return NULL;

    clusters[0] = (float *)malloc((size_t)nclusters * (size_t)nfeatures * sizeof(float));
    if (!clusters[0]) {
        free(clusters);
        return NULL;
    }

    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    for (i = 0; i < nclusters; i++) {
        float *restrict c = clusters[i];
        float *restrict f = feature[n];
        for (j = 0; j < nfeatures; j++)
            c[j] = f[j];
        n++;
    }

    for (i = 0; i < npoints; i++)
        membership[i] = -1;

    new_centers_len = (int *)calloc((size_t)nclusters, sizeof(int));
    if (!new_centers_len) {
        free(clusters[0]);
        free(clusters);
        return NULL;
    }

    new_centers = (float **)malloc((size_t)nclusters * sizeof(float *));
    if (!new_centers) {
        free(new_centers_len);
        free(clusters[0]);
        free(clusters);
        return NULL;
    }

    new_centers[0] = (float *)calloc((size_t)nclusters * (size_t)nfeatures, sizeof(float));
    if (!new_centers[0]) {
        free(new_centers);
        free(new_centers_len);
        free(clusters[0]);
        free(clusters);
        return NULL;
    }

    for (i = 1; i < nclusters; i++)
        new_centers[i] = new_centers[i - 1] + nfeatures;

    do {
        delta = 0.0f;

        for (i = 0; i < npoints; i++) {
            float *restrict feat_i = feature[i];
            int old_index = membership[i];

            index = find_nearest_point(feat_i, nfeatures, clusters, nclusters);

            if (old_index != index)
                delta += 1.0f;

            membership[i] = index;

            new_centers_len[index]++;
            {
                float *restrict nc = new_centers[index];
                int jf;
                for (jf = 0; jf < nfeatures; jf++)
                    nc[jf] += feat_i[jf];
            }
        }

        for (i = 0; i < nclusters; i++) {
            int len = new_centers_len[i];
            float *restrict c = clusters[i];
            float *restrict nc = new_centers[i];

            if (len > 0) {
                float inv_len = 1.0f / (float)len;
                for (j = 0; j < nfeatures; j++) {
                    c[j] = nc[j] * inv_len;
                    nc[j] = 0.0f;
                }
            } else {
                for (j = 0; j < nfeatures; j++)
                    nc[j] = 0.0f;
            }
            new_centers_len[i] = 0;
        }

    } while (delta > threshold && loop++ < 500);

    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}
