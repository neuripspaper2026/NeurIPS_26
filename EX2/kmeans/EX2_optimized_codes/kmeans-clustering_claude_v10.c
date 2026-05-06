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

int find_nearest_point(float *pt,                  /* [nfeatures] */
                       int nfeatures, float **pts, /* [npts][nfeatures] */
                       int npts) {
    int index = 0, i;
    float min_dist = FLT_MAX;

    /* find the cluster center id with min distance to pt */
    for (i = 0; i < npts; i++) {
        float dist;
        dist = euclid_dist_2(pt, pts[i], nfeatures); /* no need square root */
        if (dist < min_dist) {
            min_dist = dist;
            index = i;
        }
    }
    return (index);
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* multi-dimensional spatial Euclid distance square */
__inline float euclid_dist_2(float *pt1, float *pt2, int numdims) {
    int i;
    float ans = 0.0;

    for (i = 0; i < numdims; i++)
        ans += (pt1[i] - pt2[i]) * (pt1[i] - pt2[i]);

    return (ans);
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
    clusters = (float **)malloc(nclusters * sizeof(float *));
    clusters[0] = (float *)malloc(nclusters * nfeatures * sizeof(float));
    for (i = 1; i < nclusters; i++)
        clusters[i] = clusters[i - 1] + nfeatures;

    /* randomly pick cluster centers */
    for (i = 0; i < nclusters; i++) {
        // n = (int)rand() % npoints;
        for (j = 0; j < nfeatures; j++)
            clusters[i][j] = feature[n][j];
        n++;
    }

    for (i = 0; i < npoints; i++)
        membership[i] = -1;

    /* need to initialize new_centers_len and new_centers[0] to all 0 */
    new_centers_len = (int *)calloc(nclusters, sizeof(int));

    new_centers = (float **)malloc(nclusters * sizeof(float *));
    new_centers[0] = (float *)calloc(nclusters * nfeatures, sizeof(float));
    for (i = 1; i < nclusters; i++)
        new_centers[i] = new_centers[i - 1] + nfeatures;

#ifdef _OPENMP
    int num_threads = omp_get_max_threads();
    float **local_new_centers = (float **)malloc(num_threads * sizeof(float *));
    int **local_new_centers_len = (int **)malloc(num_threads * sizeof(int *));
    
    for (i = 0; i < num_threads; i++) {
        local_new_centers[i] = (float *)calloc(nclusters * nfeatures, sizeof(float));
        local_new_centers_len[i] = (int *)calloc(nclusters, sizeof(int));
    }
#endif

    do {
        delta = 0.0;
        
#ifdef _OPENMP
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            float *my_new_centers = local_new_centers[tid];
            int *my_new_centers_len = local_new_centers_len[tid];
            float local_delta = 0.0;
            int local_index;
            
            #pragma omp for schedule(static) nowait
            for (i = 0; i < npoints; i++) {
                /* find the index of nearest cluster centers */
                local_index = find_nearest_point(feature[i], nfeatures, clusters, nclusters);
                
                /* if membership changes, increase delta by 1 */
                if (membership[i] != local_index)
                    local_delta += 1.0;

                /* assign the membership to object i */
                membership[i] = local_index;

                /* update new cluster centers : sum of all objects located within */
                my_new_centers_len[local_index]++;
                for (j = 0; j < nfeatures; j++)
                    my_new_centers[local_index * nfeatures + j] += feature[i][j];
            }
            
            #pragma omp atomic
            delta += local_delta;
            
            #pragma omp barrier
            
            #pragma omp for schedule(static)
            for (i = 0; i < nclusters; i++) {
                for (int t = 0; t < num_threads; t++) {
                    new_centers_len[i] += local_new_centers_len[t][i];
                    for (j = 0; j < nfeatures; j++) {
                        new_centers[i][j] += local_new_centers[t][i * nfeatures + j];
                    }
                }
            }
        }
#else
        for (i = 0; i < npoints; i++) {
            /* find the index of nearest cluster centers */
            index = find_nearest_point(feature[i], nfeatures, clusters, nclusters);
            
            /* if membership changes, increase delta by 1 */
            if (membership[i] != index)
                delta += 1.0;

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
            for (j = 0; j < nfeatures; j++) {
                if (new_centers_len[i] > 0)
                    clusters[i][j] = new_centers[i][j] / new_centers_len[i];
                new_centers[i][j] = 0.0; /* set back to 0 */
            }
            new_centers_len[i] = 0; /* set back to 0 */
        }

#ifdef _OPENMP
        for (i = 0; i < num_threads; i++) {
            for (j = 0; j < nclusters * nfeatures; j++) {
                local_new_centers[i][j] = 0.0;
            }
            for (j = 0; j < nclusters; j++) {
                local_new_centers_len[i][j] = 0;
            }
        }
#endif

    } while (delta > threshold && loop++ < 500);

#ifdef _OPENMP
    for (i = 0; i < num_threads; i++) {
        free(local_new_centers[i]);
        free(local_new_centers_len[i]);
    }
    free(local_new_centers);
    free(local_new_centers_len);
#endif

    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}
