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

    int i, j, n = 0, loop = 0;
    int *new_centers_len; /* [nclusters]: no. of points in each cluster */
    float **new_centers;  /* [nclusters][nfeatures] */
    float **clusters;     /* out: [nclusters][nfeatures] */
    float delta;

    /* allocate space for returning variable clusters[] */
    clusters = (float **)malloc((size_t)nclusters * sizeof(float *));
    clusters[0] = (float *)malloc((size_t)nclusters * (size_t)nfeatures * sizeof(float));
    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers */
    for (i = 0; i < nclusters; i++) {
        for (j = 0; j < nfeatures; j++)
            clusters[i][j] = feature[n][j];
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

#ifdef _OPENMP
    /* Precompute constants for OpenMP loop scheduling */
    const int fn = nfeatures;
#endif

    do {
        delta = 0.0f;

#ifdef _OPENMP
#pragma omp parallel
        {
            int *local_new_centers_len;
            float **local_new_centers;
            float local_delta = 0.0f;
            int tid, nthreads;

            nthreads = omp_get_num_threads();
            tid = omp_get_thread_num();

#pragma omp single
            {
                /* allocate per-thread accumulators once per iteration */
                /* layout: [nthreads][nclusters] for lengths */
            }
#pragma omp barrier

            /* Allocate TLS accumulators on first entry to parallel region */
#pragma omp single
            {
                /* nothing here: handled outside of parallel region in serial code */
            }

            /* Each thread uses thread-private accumulators on the stack for speed */
            local_new_centers_len = (int *)calloc((size_t)nclusters, sizeof(int));
            local_new_centers = (float **)malloc((size_t)nclusters * sizeof(float *));
            local_new_centers[0] = (float *)calloc((size_t)nclusters * (size_t)fn, sizeof(float));
            for (i = 1; i < nclusters; i++)
                local_new_centers[i] = local_new_centers[i - 1] + fn;

#pragma omp for schedule(static)
            for (i = 0; i < npoints; i++) {
                int index = 0;
                float min_dist = FLT_MAX;

                /* Manual inlining of find_nearest_point for better locality and vectorization */
                float *pt = feature[i];
                for (j = 0; j < nclusters; j++) {
                    float *cl = clusters[j];
                    float dist = 0.0f;
                    for (int d = 0; d < fn; d++) {
                        float diff = pt[d] - cl[d];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        index = j;
                    }
                }

                if (membership[i] != index)
                    local_delta += 1.0f;

                membership[i] = index;

                local_new_centers_len[index]++;
                float *dst = local_new_centers[index];
                float *src = pt;
                for (j = 0; j < fn; j++)
                    dst[j] += src[j];
            }

#pragma omp critical
            {
                delta += local_delta;
                for (i = 0; i < nclusters; i++) {
                    new_centers_len[i] += local_new_centers_len[i];
                    float *dst = new_centers[i];
                    float *src = local_new_centers[i];
                    for (j = 0; j < fn; j++)
                        dst[j] += src[j];
                }
            }

            free(local_new_centers[0]);
            free(local_new_centers);
            free(local_new_centers_len);
        }
#else
        for (i = 0; i < npoints; i++) {
            int index = find_nearest_point(feature[i], nfeatures, clusters, nclusters);

            if (membership[i] != index)
                delta += 1.0f;

            membership[i] = index;

            new_centers_len[index]++;
            for (j = 0; j < nfeatures; j++)
                new_centers[index][j] += feature[i][j];
        }
#endif

        /* replace old cluster centers with new_centers */
#ifdef _OPENMP
#pragma omp parallel for private(j) schedule(static)
#endif
        for (i = 0; i < nclusters; i++) {
            int count = new_centers_len[i];
            if (count > 0) {
                float inv_count = 1.0f / (float)count;
                float *cluster_i = clusters[i];
                float *new_center_i = new_centers[i];
                for (j = 0; j < nfeatures; j++) {
                    cluster_i[j] = new_center_i[j] * inv_count;
                    new_center_i[j] = 0.0f;
                }
            } else {
                float *new_center_i = new_centers[i];
                for (j = 0; j < nfeatures; j++)
                    new_center_i[j] = 0.0f;
            }
            new_centers_len[i] = 0;
        }

    } while (delta > threshold && loop++ < 500);


    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}
