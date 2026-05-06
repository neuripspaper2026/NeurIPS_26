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

#pragma omp simd reduction(+:ans)
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

        /* parallel assignment step */
#ifdef _OPENMP
#pragma omp parallel
        {
            int *local_new_centers_len = (int *)calloc((size_t)nclusters, sizeof(int));
            float *local_new_centers_blk =
                (float *)calloc((size_t)nclusters * (size_t)nfeatures, sizeof(float));

#pragma omp for reduction(+:delta) schedule(static)
            for (i = 0; i < npoints; i++) {
                int local_index;
                float *feat_i = feature[i];

                /* find the index of nearest cluster center */
                local_index = find_nearest_point(feat_i, nfeatures, clusters, nclusters);

                /* if membership changes, increase delta by 1 */
                if (membership[i] != local_index)
                    delta += 1.0f;

                /* assign the membership to object i */
                membership[i] = local_index;

                /* update local cluster centers : sum of all objects located within */
                local_new_centers_len[local_index]++;
                {
                    float *dst = local_new_centers_blk + (size_t)local_index * (size_t)nfeatures;
#pragma omp simd
                    for (j = 0; j < nfeatures; j++) {
                        dst[j] += feat_i[j];
                    }
                }
            }

            /* combine local partial sums into global accumulators */
#pragma omp for schedule(static)
            for (i = 0; i < nclusters; i++) {
                int len = local_new_centers_len[i];
                if (len != 0) {
#pragma omp atomic
                    new_centers_len[i] += len;
                }

                float *src = local_new_centers_blk + (size_t)i * (size_t)nfeatures;
                float *dst = new_centers[i];
#pragma omp simd
                for (j = 0; j < nfeatures; j++) {
#pragma omp atomic
                    dst[j] += src[j];
                }
            }

            free(local_new_centers_len);
            free(local_new_centers_blk);
        }
#else
        for (i = 0; i < npoints; i++) {
            float *feat_i = feature[i];

            /* find the index of nearest cluster centers */
            index = find_nearest_point(feat_i, nfeatures, clusters, nclusters);

            /* if membership changes, increase delta by 1 */
            if (membership[i] != index)
                delta += 1.0f;

            /* assign the membership to object i */
            membership[i] = index;

            /* update new cluster centers : sum of all objects located within */
            new_centers_len[index]++;
            {
                float *dst = new_centers[index];
#pragma omp simd
                for (j = 0; j < nfeatures; j++)
                    dst[j] += feat_i[j];
            }
        }
#endif

        /* replace old cluster centers with new_centers */
        for (i = 0; i < nclusters; i++) {
            int len = new_centers_len[i];
            float inv_len = (len > 0) ? (1.0f / (float)len) : 0.0f;
            float *c = clusters[i];
            float *nc = new_centers[i];

#pragma omp simd
            for (j = 0; j < nfeatures; j++) {
                if (len > 0)
                    c[j] = nc[j] * inv_len;
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
