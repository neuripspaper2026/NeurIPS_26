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

    /* simple, cache-friendly inner loop; compiler can auto-vectorize */
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

    /* randomly pick cluster centers (here, sequential positions) */
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
        /* Parallel assignment and local accumulation to reduce contention */
        #pragma omp parallel
        {
            int *local_new_centers_len = (int *)calloc((size_t)nclusters, sizeof(int));
            float *local_new_centers = (float *)calloc((size_t)nclusters * (size_t)nfeatures, sizeof(float));
            float local_delta = 0.0f;

            #pragma omp for nowait
            for (i = 0; i < npoints; i++) {
                /* find the index of nearest cluster center */
                index = find_nearest_point(feature[i], nfeatures, clusters,
                                           nclusters);

                /* if membership changes, increase delta by 1 */
                if (membership[i] != index)
                    local_delta += 1.0f;

                /* assign the membership to object i */
                membership[i] = index;

                /* update new cluster centers : sum of all objects located within */
                local_new_centers_len[index]++;
                {
                    float *restrict dst = &local_new_centers[index * nfeatures];
                    float *restrict src = feature[i];
                    for (j = 0; j < nfeatures; j++)
                        dst[j] += src[j];
                }
            }

            /* reduce thread-local accumulators into global ones */
            #pragma omp critical
            {
                int c, f;
                for (c = 0; c < nclusters; c++) {
                    new_centers_len[c] += local_new_centers_len[c];
                    float *restrict gdst = new_centers[c];
                    float *restrict gsrc = &local_new_centers[c * nfeatures];
                    for (f = 0; f < nfeatures; f++)
                        gdst[f] += gsrc[f];
                }
                delta += local_delta;
            }

            free(local_new_centers_len);
            free(local_new_centers);
        }
#else
        /* Serial version */
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
            for (j = 0; j < nfeatures; j++)
                new_centers[index][j] += feature[i][j];
        }
#endif

        /* replace old cluster centers with new_centers */
        for (i = 0; i < nclusters; i++) {
            int len = new_centers_len[i];
            if (len > 0) {
                float inv_len = 1.0f / (float)len;
                float *restrict cptr = clusters[i];
                float *restrict nptr = new_centers[i];
                for (j = 0; j < nfeatures; j++) {
                    cptr[j] = nptr[j] * inv_len;
                    nptr[j] = 0.0f; /* set back to 0 */
                }
            } else {
                float *restrict nptr = new_centers[i];
                for (j = 0; j < nfeatures; j++) {
                    nptr[j] = 0.0f; /* set back to 0 */
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
