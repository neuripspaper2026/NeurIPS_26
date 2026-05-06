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
    float ans = 0.0f;
    int i = 0;

    /* unrolled loop for better ILP and fewer loop overheads */
    int limit = numdims & ~3; /* largest multiple of 4 <= numdims */
    for (; i < limit; i += 4) {
        float d0 = pt1[i]     - pt2[i];
        float d1 = pt1[i + 1] - pt2[i + 1];
        float d2 = pt1[i + 2] - pt2[i + 2];
        float d3 = pt1[i + 3] - pt2[i + 3];
        ans += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    for (; i < numdims; i++) {
        float d = pt1[i] - pt2[i];
        ans += d * d;
    }

    return ans;
}

int find_nearest_point(float *pt,                  /* [nfeatures] */
                       int nfeatures, float **pts, /* [npts][nfeatures] */
                       int npts) {
    int index = 0;
    int i;
    float min_dist = FLT_MAX;

    for (i = 0; i < npts; i++) {
        float dist = euclid_dist_2(pt, pts[i], nfeatures); /* squared distance */
        if (dist < min_dist) {
            min_dist = dist;
            index = i;
        }
    }
    return index;
}

/*----< kmeans_clustering() >---------------------------------------------*/
float **kmeans_clustering(float **feature, /* in: [npoints][nfeatures] */
                          int nfeatures, int npoints, int nclusters,
                          float threshold, int *membership) /* out: [npoints] */
{

    int i, j, n = 0, index, loop = 0;
    int *new_centers_len; /* [nclusters]: no. of points in each cluster */
    float **new_centers;  /* [nclusters][nfeatures] */
    float **clusters;     /* out: [nclusters][nfeatures] */
    float delta;

    clusters = (float **)malloc((size_t)nclusters * sizeof(float *));
    clusters[0] = (float *)malloc((size_t)nclusters * (size_t)nfeatures * sizeof(float));
    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    for (i = 0; i < nclusters; i++) {
        for (j = 0; j < nfeatures; j++)
            clusters[i][j] = feature[n][j];
        n++;
    }

    for (i = 0; i < npoints; i++)
        membership[i] = -1;

    new_centers_len = (int *)calloc((size_t)nclusters, sizeof(int));

    new_centers = (float **)malloc((size_t)nclusters * sizeof(float *));
    new_centers[0] = (float *)calloc((size_t)nclusters * (size_t)nfeatures, sizeof(float));
    for (i = 1; i < nclusters; i++)
        new_centers[i] = new_centers[i - 1] + nfeatures;

    do {
        delta = 0.0f;

        for (i = 0; i < npoints; i++) {
            float *feat_i = feature[i];

            index = find_nearest_point(feat_i, nfeatures, clusters, nclusters);

            if (membership[i] != index)
                delta += 1.0f;

            membership[i] = index;

            new_centers_len[index]++;
            {
                float *nc = new_centers[index];
                int limit = nfeatures & ~3;
                int k = 0;

                for (; k < limit; k += 4) {
                    nc[k]     += feat_i[k];
                    nc[k + 1] += feat_i[k + 1];
                    nc[k + 2] += feat_i[k + 2];
                    nc[k + 3] += feat_i[k + 3];
                }
                for (; k < nfeatures; k++)
                    nc[k] += feat_i[k];
            }
        }

        for (i = 0; i < nclusters; i++) {
            float *cl = clusters[i];
            float *nc = new_centers[i];
            int len = new_centers_len[i];

            if (len > 0) {
                float inv_len = 1.0f / (float)len;
                int limit = nfeatures & ~3;
                int k = 0;

                for (; k < limit; k += 4) {
                    cl[k]     = nc[k]     * inv_len;
                    cl[k + 1] = nc[k + 1] * inv_len;
                    cl[k + 2] = nc[k + 2] * inv_len;
                    cl[k + 3] = nc[k + 3] * inv_len;

                    nc[k]     = 0.0f;
                    nc[k + 1] = 0.0f;
                    nc[k + 2] = 0.0f;
                    nc[k + 3] = 0.0f;
                }
                for (; k < nfeatures; k++) {
                    cl[k] = nc[k] * inv_len;
                    nc[k] = 0.0f;
                }
            } else {
                int k;
                int limit = nfeatures & ~3;
                int k2 = 0;
                for (; k2 < limit; k2 += 4) {
                    nc[k2]     = 0.0f;
                    nc[k2 + 1] = 0.0f;
                    nc[k2 + 2] = 0.0f;
                    nc[k2 + 3] = 0.0f;
                }
                for (k = k2; k < nfeatures; k++)
                    nc[k] = 0.0f;
            }

            new_centers_len[i] = 0;
        }

    } while (delta > threshold && loop++ < 500);

    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}
