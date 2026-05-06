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

    /* simple inner-product style loop, encourages auto-vectorization */
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
    if (!clusters) {
        return NULL;
    }
    clusters[0] =
        (float *)malloc((size_t)nclusters * (size_t)nfeatures * sizeof(float));
    if (!clusters[0]) {
        free(clusters);
        return NULL;
    }
    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers (here: first nclusters points in order) */
    for (i = 0; i < nclusters; i++) {
        for (j = 0; j < nfeatures; j++)
            clusters[i][j] = feature[n][j];
        n++;
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
    new_centers[0] = (float *)calloc((size_t)nclusters * (size_t)nfeatures,
                                     sizeof(float));
    if (!new_centers[0]) {
        free(new_centers);
        free(new_centers_len);
        free(clusters[0]);
        free(clusters);
        return NULL;
    }
    for (i = 1; i < nclusters; i++)
        new_centers[i] = new_centers[i - 1] + nfeatures;

#ifdef _OPENMP
    /* Parallelized K-Means main loop */
    do {
        delta = 0.0f;

        /* per-thread temporaries to avoid false sharing and atomic ops */
        int nthreads = 1;
#pragma omp parallel
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
            nthreads = omp_get_num_threads();
#endif
            /* Allocate per-thread accumulators on first parallel entry */
#pragma omp single
            {
                (void)tid;
                (void)nthreads;
            }
        }

        /* Allocate per-thread accumulators outside main parallel for reuse */
        int max_threads = omp_get_max_threads();
        int *local_new_centers_len =
            (int *)calloc((size_t)max_threads * (size_t)nclusters,
                          sizeof(int));
        float *local_new_centers =
            (float *)calloc((size_t)max_threads * (size_t)nclusters *
                                (size_t)nfeatures,
                            sizeof(float));

        /* assign points to clusters and compute partial sums */
#pragma omp parallel default(none)                                               \
    shared(feature, membership, clusters, new_centers_len, new_centers,         \
           npoints, nfeatures, nclusters, threshold, max_threads,               \
           local_new_centers_len, local_new_centers, delta, loop)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            int *len_thread =
                &local_new_centers_len[tid * nclusters];
            float *centers_thread =
                &local_new_centers[tid * nclusters * nfeatures];

#pragma omp for reduction(+ : delta) schedule(static)
            for (int ii = 0; ii < npoints; ii++) {
                int local_index = find_nearest_point(
                    feature[ii], nfeatures, clusters, nclusters);

                if (membership[ii] != local_index)
                    delta += 1.0f;

                membership[ii] = local_index;

                len_thread[local_index]++;
                float *restrict ctr =
                    &centers_thread[local_index * nfeatures];
                float *restrict feat = feature[ii];
                for (int jj = 0; jj < nfeatures; jj++)
                    ctr[jj] += feat[jj];
            }
        } /* end parallel */

        /* Reduce per-thread partial sums into global new_centers */
        for (i = 0; i < nclusters; i++) {
            int len_sum = 0;
            for (int t = 0; t < max_threads; t++) {
                len_sum += local_new_centers_len[t * nclusters + i];
            }
            new_centers_len[i] = len_sum;

            float *dst = new_centers[i];
            for (j = 0; j < nfeatures; j++) {
                float sum_val = 0.0f;
                for (int t = 0; t < max_threads; t++) {
                    sum_val += local_new_centers
                               [t * nclusters * nfeatures +
                                i * nfeatures + j];
                }
                dst[j] = sum_val;
            }
        }

        free(local_new_centers_len);
        free(local_new_centers);

        /* replace old cluster centers with new_centers */
#pragma omp parallel for default(none) private(j) schedule(static)              \
    shared(clusters, new_centers, new_centers_len, nclusters, nfeatures)
        for (i = 0; i < nclusters; i++) {
            int len = new_centers_len[i];
            float *restrict cl = clusters[i];
            float *restrict nc = new_centers[i];
            if (len > 0) {
                float inv_len = 1.0f / (float)len;
                for (j = 0; j < nfeatures; j++)
                    cl[j] = nc[j] * inv_len;
            }
            for (j = 0; j < nfeatures; j++)
                nc[j] = 0.0f;
            new_centers_len[i] = 0;
        }

    } while (delta > threshold && loop++ < 500);
#else
    /* Serial K-Means main loop (optimized but no OpenMP) */
    do {
        delta = 0.0f;

        for (i = 0; i < npoints; i++) {
            /* find the index of nearest cluster center */
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
            float *restrict feat = feature[i];
            for (j = 0; j < nfeatures; j++)
                nc[j] += feat[j];
        }

        /* replace old cluster centers with new_centers */
        for (i = 0; i < nclusters; i++) {
            int len = new_centers_len[i];
            float *restrict cl = clusters[i];
            float *restrict nc = new_centers[i];
            if (len > 0) {
                float inv_len = 1.0f / (float)len;
                for (j = 0; j < nfeatures; j++)
                    cl[j] = nc[j] * inv_len;
            }
            for (j = 0; j < nfeatures; j++)
                nc[j] = 0.0f; /* set back to 0 */
            new_centers_len[i] = 0; /* set back to 0 */
        }

    } while (delta > threshold && loop++ < 500);
#endif

    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}
