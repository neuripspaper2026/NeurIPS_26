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

    /* unrolled loop for better ILP and reduced loop overhead */
    int limit = numdims & ~3; /* largest multiple of 4 <= numdims */
    for (; i < limit; i += 4) {
        float d0 = pt1[i]     - pt2[i];
        float d1 = pt1[i + 1] - pt2[i + 1];
        float d2 = pt1[i + 2] - pt2[i + 2];
        float d3 = pt1[i + 3] - pt2[i + 3];
        ans += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    /* handle remaining elements */
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

    /* find the cluster center id with min distance to pt */
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

    /* allocate space for returning variable clusters[] */
    clusters = (float **)malloc((size_t)nclusters * sizeof(float *));
    clusters[0] = (float *)malloc((size_t)nclusters * (size_t)nfeatures * sizeof(float));
    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers (deterministic sequence here) */
    for (i = 0; i < nclusters; i++) {
        float *restrict c = clusters[i];
        float *restrict f = feature[n];
        for (j = 0; j < nfeatures; j++)
            c[j] = f[j];
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
            float *restrict feat_i = feature[i];

            /* find the index of nearest cluster centers */
            index = find_nearest_point(feat_i, nfeatures, clusters,
                                       nclusters);
            /* if membership changes, increase delta by 1 */
            if (membership[i] != index)
                delta += 1.0f;

            /* assign the membership to object i */
            membership[i] = index;

            /* update new cluster centers : sum of all objects located within */
            new_centers_len[index]++;
            {
                float *restrict nc = new_centers[index];
                int jf = 0;
                int limitf = nfeatures & ~3;

                /* unrolled accumulation */
                for (; jf < limitf; jf += 4) {
                    nc[jf]     += feat_i[jf];
                    nc[jf + 1] += feat_i[jf + 1];
                    nc[jf + 2] += feat_i[jf + 2];
                    nc[jf + 3] += feat_i[jf + 3];
                }
                for (; jf < nfeatures; jf++) {
                    nc[jf] += feat_i[jf];
                }
            }
        }

        /* replace old cluster centers with new_centers */
        for (i = 0; i < nclusters; i++) {
            float len = (float)new_centers_len[i];
            float *restrict c = clusters[i];
            float *restrict nc = new_centers[i];

            if (len > 0.0f) {
                float inv_len = 1.0f / len;
                int jf = 0;
                int limitf = nfeatures & ~3;

                /* unrolled normalization and reset */
                for (; jf < limitf; jf += 4) {
                    c[jf]     = nc[jf]     * inv_len;
                    c[jf + 1] = nc[jf + 1] * inv_len;
                    c[jf + 2] = nc[jf + 2] * inv_len;
                    c[jf + 3] = nc[jf + 3] * inv_len;

                    nc[jf]     = 0.0f;
                    nc[jf + 1] = 0.0f;
                    nc[jf + 2] = 0.0f;
                    nc[jf + 3] = 0.0f;
                }
                for (; jf < nfeatures; jf++) {
                    c[jf] = nc[jf] * inv_len;
                    nc[jf] = 0.0f;
                }
            } else {
                /* no points assigned: just reset accumulators */
                int jf = 0;
                int limitf = nfeatures & ~3;
                for (; jf < limitf; jf += 4) {
                    nc[jf]     = 0.0f;
                    nc[jf + 1] = 0.0f;
                    nc[jf + 2] = 0.0f;
                    nc[jf + 3] = 0.0f;
                }
                for (; jf < nfeatures; jf++) {
                    nc[jf] = 0.0f;
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
