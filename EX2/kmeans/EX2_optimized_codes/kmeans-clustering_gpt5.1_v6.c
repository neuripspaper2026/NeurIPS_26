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

    /* simple loop with possible compiler auto-vectorization */
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

    /* randomly pick cluster centers (here sequential from feature[0..]) */
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

        /* Parallel assignment and accumulation over points */
        #ifdef _OPENMP
        #pragma omp parallel default(none) \
            shared(feature, clusters, membership, new_centers, new_centers_len, npoints, nfeatures, nclusters, threshold) \
            reduction(+:delta)
        {
            int i_local, j_local;

            /* Thread-local accumulators to reduce contention on shared arrays */
            float *local_new_centers = (float *)calloc((size_t)nclusters * (size_t)nfeatures, sizeof(float));
            int   *local_new_centers_len = (int *)calloc((size_t)nclusters, sizeof(int));

            #pragma omp for schedule(static)
            for (i_local = 0; i_local < npoints; i_local++) {
                int index_local = find_nearest_point(feature[i_local], nfeatures, clusters,
                                                     nclusters);
                if (membership[i_local] != index_local)
                    delta += 1.0f;

                membership[i_local] = index_local;

                local_new_centers_len[index_local]++;
                {
                    float *dst = &local_new_centers[index_local * nfeatures];
                    float *src = feature[i_local];
                    for (j_local = 0; j_local < nfeatures; j_local++)
                        dst[j_local] += src[j_local];
                }
            }

            /* Reduce thread-local accumulators into shared new_centers/new_centers_len */
            #pragma omp for schedule(static)
            for (i_local = 0; i_local < nclusters; i_local++) {
                int base = i_local * nfeatures;
                new_centers_len[i_local] += local_new_centers_len[i_local];
                for (j_local = 0; j_local < nfeatures; j_local++) {
                    new_centers[i_local][j_local] += local_new_centers[base + j_local];
                }
            }

            free(local_new_centers);
            free(local_new_centers_len);
        }
        #else
        for (i = 0; i < npoints; i++) {
            index = find_nearest_point(feature[i], nfeatures, clusters,
                                       nclusters);

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
        #pragma omp parallel for default(none) private(j) shared(clusters, new_centers, new_centers_len, nclusters, nfeatures)
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
