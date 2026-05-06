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


    do {
        delta = 0.0f;

#ifdef _OPENMP
        /* Parallel assignment and accumulation into per-thread partial centers */
        int nthreads = 1;
#pragma omp parallel
        {
            int tid;
#ifdef _OPENMP
            tid = omp_get_thread_num();
            nthreads = omp_get_num_threads();
#else
            tid = 0;
#endif

            /* Allocate per-thread partial sums and counts on first thread only,
               then share via shared pointers - done outside loop to avoid malloc
               in parallel region, but for simplicity and to avoid new functions,
               we use static-size OpenMP shared arrays via dynamic allocation
               guarded by single. */
        }
        /* Allocate per-thread temporary accumulators */
        int t;
        int **t_new_centers_len = (int **)malloc((size_t)nthreads * sizeof(int *));
        float **t_new_centers = (float **)malloc((size_t)nthreads * sizeof(float *));
        for (t = 0; t < nthreads; t++) {
            t_new_centers_len[t] = (int *)calloc((size_t)nclusters, sizeof(int));
            t_new_centers[t] = (float *)calloc((size_t)nclusters * (size_t)nfeatures, sizeof(float));
        }

#pragma omp parallel default(none) shared(feature, clusters, membership, npoints, nfeatures, nclusters, threshold, t_new_centers_len, t_new_centers, delta, nthreads)
        {
            int i_local, j_local;
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            float local_delta = 0.0f;
            int *local_new_centers_len = t_new_centers_len[tid];
            float *local_new_centers_flat = t_new_centers[tid];

#pragma omp for nowait
            for (i_local = 0; i_local < npoints; i_local++) {
                float *pt = feature[i_local];
                int best_index = 0;
                float min_dist = FLT_MAX;
                int c;
                for (c = 0; c < nclusters; c++) {
                    float dist = 0.0f;
                    float *cluster_pt = clusters[c];
                    int d;
                    for (d = 0; d < nfeatures; d++) {
                        float diff = pt[d] - cluster_pt[d];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_index = c;
                    }
                }

                if (membership[i_local] != best_index)
                    local_delta += 1.0f;
                membership[i_local] = best_index;

                local_new_centers_len[best_index]++;
                float *center_row = local_new_centers_flat + (size_t)best_index * (size_t)nfeatures;
                for (j_local = 0; j_local < nfeatures; j_local++) {
                    center_row[j_local] += pt[j_local];
                }
            }

#pragma omp atomic
            delta += local_delta;
        }

        /* Reduce per-thread partial centers into global new_centers/new_centers_len */
        for (t = 0; t < nthreads; t++) {
            int *len_t = t_new_centers_len[t];
            float *cent_t = t_new_centers[t];
            for (i = 0; i < nclusters; i++) {
                new_centers_len[i] += len_t[i];
                float *dst = new_centers[i];
                float *src = cent_t + (size_t)i * (size_t)nfeatures;
                for (j = 0; j < nfeatures; j++) {
                    dst[j] += src[j];
                }
            }
        }

        /* free per-thread temporaries */
        for (t = 0; t < nthreads; t++) {
            free(t_new_centers_len[t]);
            free(t_new_centers[t]);
        }
        free(t_new_centers_len);
        free(t_new_centers);
#else
        /* Serial version */
        for (i = 0; i < npoints; i++) {
            index = find_nearest_point(feature[i], nfeatures, clusters,
                                       nclusters);
            if (membership[i] != index)
                delta += 1.0f;

            membership[i] = index;

            new_centers_len[index]++;
            float *nc = new_centers[index];
            float *pt = feature[i];
            for (j = 0; j < nfeatures; j++)
                nc[j] += pt[j];
        }
#endif

        /* replace old cluster centers with new_centers */
        for (i = 0; i < nclusters; i++) {
            int len = new_centers_len[i];
            float *cluster_row = clusters[i];
            float *new_center_row = new_centers[i];
            if (len > 0) {
                float inv_len = 1.0f / (float)len;
                for (j = 0; j < nfeatures; j++) {
                    cluster_row[j] = new_center_row[j] * inv_len;
                    new_center_row[j] = 0.0f;
                }
            } else {
                for (j = 0; j < nfeatures; j++) {
                    new_center_row[j] = 0.0f;
                }
            }
            new_centers_len[i] = 0;
        }

    } while (delta > threshold && loop++ < 500);


    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}
