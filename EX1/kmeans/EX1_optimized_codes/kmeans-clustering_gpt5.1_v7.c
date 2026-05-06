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
    int i;
    float ans = 0.0f;

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

    /* randomly pick cluster centers (here: first nclusters points) */
    for (i = 0; i < nclusters; i++) {
        float *restrict dst = clusters[i];
        float *restrict src = feature[n];
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
            {
                float *restrict nc = new_centers[index];
                float *restrict ft = feature[i];
                for (j = 0; j < nfeatures; j++)
                    nc[j] += ft[j];
            }
        }

        /* replace old cluster centers with new_centers */
        for (i = 0; i < nclusters; i++) {
            const int len = new_centers_len[i];
            float *restrict cl = clusters[i];
            float *restrict nc = new_centers[i];

            if (len > 0) {
                const float inv_len = 1.0f / (float)len;
                for (j = 0; j < nfeatures; j++) {
                    cl[j] = nc[j] * inv_len;
                    nc[j] = 0.0f; /* set back to 0 */
                }
            } else {
                for (j = 0; j < nfeatures; j++) {
                    nc[j] = 0.0f; /* set back to 0 */
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
