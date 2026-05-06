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
    int i;

    for (i = 0; i < numdims; i++) {
        float diff = pt1[i] - pt2[i];
        ans += diff * diff;
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
        float dist = euclid_dist_2(pt, pts[i], nfeatures); /* no need square root */
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
            float *feature_i = feature[i];

            index = find_nearest_point(feature_i, nfeatures, clusters, nclusters);

            if (membership[i] != index)
                delta += 1.0f;

            membership[i] = index;

            new_centers_len[index]++;
            {
                float *restrict nc = new_centers[index];
                int nf = nfeatures;
                int j_local;
                for (j_local = 0; j_local < nf; j_local++)
                    nc[j_local] += feature_i[j_local];
            }
        }

        for (i = 0; i < nclusters; i++) {
            float *restrict ci = clusters[i];
            float *restrict nci = new_centers[i];
            int len = new_centers_len[i];

            if (len > 0) {
                float inv_len = 1.0f / (float)len;
                for (j = 0; j < nfeatures; j++)
                    ci[j] = nci[j] * inv_len;
            }

            for (j = 0; j < nfeatures; j++)
                nci[j] = 0.0f;

            new_centers_len[i] = 0;
        }

    } while (delta > threshold && loop++ < 500);

    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}
