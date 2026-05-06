#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
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
    int index = 0, i;
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
    if (!clusters)
        return NULL;
    clusters[0] = (float *)malloc((size_t)nclusters * (size_t)nfeatures * sizeof(float));
    if (!clusters[0]) {
        free(clusters);
        return NULL;
    }
    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers */
    for (i = 0; i < nclusters; i++) {
        for (j = 0; j < nfeatures; j++)
            clusters[i][j] = feature[n][j];
        n++;
        if (n >= npoints)
            n = 0;
    }

    for (i = 0; i < npoints; i++)
        membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
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

        /* Parallel over points: compute nearest cluster and accumulate partial sums */
#ifdef _OPENMP
#pragma omp parallel default(none) shared(feature, clusters, membership, new_centers, new_centers_len, npoints, nclusters, nfeatures, threshold, loop) private(i, j, index)
        {
            int *local_new_centers_len = (int *)calloc((size_t)nclusters, sizeof(int));
            float *local_new_centers_buf = (float *)calloc((size_t)nclusters * (size_t)nfeatures, sizeof(float));
            float **local_new_centers = (float **)alloca((size_t)nclusters * sizeof(float *));
            int p;
            for (p = 0; p < nclusters; p++)
                local_new_centers[p] = local_new_centers_buf + (size_t)p * (size_t)nfeatures;

            float local_delta = 0.0f;

#pragma omp for schedule(static)
            for (i = 0; i < npoints; i++) {
                int nearest = 0;
                float min_dist = FLT_MAX;
                float *pt = feature[i];

                for (j = 0; j < nclusters; j++) {
                    float dist = euclid_dist_2(pt, clusters[j], nfeatures);
                    if (dist < min_dist) {
                        min_dist = dist;
                        nearest = j;
                    }
                }

                if (membership[i] != nearest)
                    local_delta += 1.0f;

                membership[i] = nearest;

                local_new_centers_len[nearest]++;
                {
                    float *dst = local_new_centers[nearest];
                    float *src = pt;
                    int d;
                    for (d = 0; d < nfeatures; d++)
                        dst[d] += src[d];
                }
            }

#pragma omp atomic
            delta += local_delta;

#pragma omp critical
            {
                for (i = 0; i < nclusters; i++) {
                    new_centers_len[i] += local_new_centers_len[i];
                    float *dst = new_centers[i];
                    float *src = local_new_centers[i];
                    for (j = 0; j < nfeatures; j++)
                        dst[j] += src[j];
                }
            }

            free(local_new_centers_len);
            free(local_new_centers_buf);
        }
#else
        for (i = 0; i < npoints; i++) {
            /* find the index of nestest cluster centers */
            index = find_nearest_point(feature[i], nfeatures, clusters,
                                       nclusters);
            /* if membership changes, increase delta by 1 */
            if (membership[i] != index)
                delta += 1.0f;

            /* assign the membership to object i */
            membership[i] = index;

            /* update new cluster centers : sum of all objects located within */
            new_centers_len[index]++;
            for (j = 0; j < nfeatures; j++)
                new_centers[index][j] += feature[i][j];
        }
#endif

        /* replace old cluster centers with new_centers */
#ifdef _OPENMP
#pragma omp parallel for default(none) shared(clusters, new_centers, new_centers_len, nclusters, nfeatures) private(i, j)
#endif
        for (i = 0; i < nclusters; i++) {
            int len = new_centers_len[i];
            if (len > 0) {
                float inv_len = 1.0f / (float)len;
                float *c = clusters[i];
                float *nc = new_centers[i];
                for (j = 0; j < nfeatures; j++) {
                    c[j] = nc[j] * inv_len;
                    nc[j] = 0.0f; /* set back to 0 */
                }
            } else {
                float *nc = new_centers[i];
                for (j = 0; j < nfeatures; j++)
                    nc[j] = 0.0f; /* set back to 0 */
            }
            new_centers_len[i] = 0; /* set back to 0 */
        }

    } while (delta > threshold && loop++ < 500);

    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}
