#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "../kmeans.h"

#define RANDOM_MAX 2147483647

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
__inline float euclid_dist_2(float *pt1, float *pt2, int numdims) {
    float ans = 0.0f;
    int i = 0;

    /* Loop unrolling by 4 for better ILP and fewer loop overheads */
    int limit = numdims & ~3;
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
    float min_dist = FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (int i = 0; i < npts; i++) {
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

    /* allocate space for returning variable clusters[] */
    clusters = (float **)malloc((size_t)nclusters * sizeof(float *));
    clusters[0] = (float *)malloc((size_t)nclusters * (size_t)nfeatures * sizeof(float));
    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers (here: first nclusters points) */
    for (i = 0; i < nclusters; i++) {
        const float *src = feature[n];
        float *dst = clusters[i];
        for (j = 0; j < nfeatures; j++)
            dst[j] = src[j];
        n++;
    }

    for (i = 0; i < npoints; i++)
        membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int *)calloc((size_t)nclusters, sizeof(int));

    new_centers = (float **)malloc((size_t)nclusters * sizeof(float *));
    new_centers[0] = (float *)calloc((size_t)nclusters * (size_t)nfeatures, sizeof(float));
    for (i = 1; i < nclusters; i++)
        new_centers[i] = new_centers[i - 1] + nfeatures;


    do {
        delta = 0.0f;
        for (i = 0; i < npoints; i++) {
            /* find the index of nearest cluster centers */
            index = find_nearest_point(feature[i], nfeatures, clusters,
                                       nclusters);
            /* if membership changes, increase delta by 1 */
            if (membership[i] != index)
                delta += 1.0f;

            /* assign the membership to object i */
            membership[i] = index;

            /* update new cluster centers : sum of all objects located within */
            new_centers_len[index]++;
            float *restrict nc = new_centers[index];
            float *restrict ft = feature[i];
            int k = 0;
            int limit = nfeatures & ~3;
            for (; k < limit; k += 4) {
                nc[k]     += ft[k];
                nc[k + 1] += ft[k + 1];
                nc[k + 2] += ft[k + 2];
                nc[k + 3] += ft[k + 3];
            }
            for (; k < nfeatures; k++) {
                nc[k] += ft[k];
            }
        }

        /* replace old cluster centers with new_centers */
        for (i = 0; i < nclusters; i++) {
            int len = new_centers_len[i];
            float inv_len = (len > 0) ? (1.0f / (float)len) : 0.0f;
            float *restrict cl = clusters[i];
            float *restrict nc = new_centers[i];

            int k = 0;
            int limit = nfeatures & ~3;
            if (len > 0) {
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
                for (; k < limit; k += 4) {
                    nc[k]     = 0.0f;
                    nc[k + 1] = 0.0f;
                    nc[k + 2] = 0.0f;
                    nc[k + 3] = 0.0f;
                }
                for (; k < nfeatures; k++) {
                    nc[k] = 0.0f;
                }
            }
            new_centers_len[i] = 0; /* set back to 0 */
        }

    } while (delta > threshold && loop++ < 500);


    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}
