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
    float **thread_new_centers = (float **)malloc(num_threads * sizeof(float *));
    thread_new_centers[0] = (float *)calloc(num_threads * nclusters * nfeatures, sizeof(float));
    for (i = 1; i < num_threads; i++)
        thread_new_centers[i] = thread_new_centers[i - 1] + nclusters * nfeatures;
    
    int **thread_new_centers_len = (int **)malloc(num_threads * sizeof(int *));
    thread_new_centers_len[0] = (int *)calloc(num_threads * nclusters, sizeof(int));
    for (i = 1; i < num_threads; i++)
        thread_new_centers_len[i] = thread_new_centers_len[i - 1] + nclusters;
#endif

    do {
        delta = 0.0;
        
#ifdef _OPENMP
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            float *my_new_centers = thread_new_centers[tid];
            int *my_new_centers_len = thread_new_centers_len[tid];
            float local_delta = 0.0;
            
            for (i = 0; i < nclusters * nfeatures; i++)
                my_new_centers[i] = 0.0;
            for (i = 0; i < nclusters; i++)
                my_new_centers_len[i] = 0;
            
            #pragma omp for schedule(static) nowait
            for (i = 0; i < npoints; i++) {
                float min_dist = FLT_MAX;
                int best_index = 0;
                
                for (int k = 0; k < nclusters; k++) {
                    float dist = 0.0;
                    for (j = 0; j < nfeatures; j++) {
                        float diff = feature[i][j] - clusters[k][j];
                        dist += diff * diff;
                    }
                    if (dist < min_dist) {
                        min_dist = dist;
                        best_index = k;
                    }
                }
                
                if (membership[i] != best_index)
                    local_delta += 1.0;
                
                membership[i] = best_index;
                
                my_new_centers_len[best_index]++;
                for (j = 0; j < nfeatures; j++)
                    my_new_centers[best_index * nfeatures + j] += feature[i][j];
            }
            
            #pragma omp atomic
            delta += local_delta;
            
            #pragma omp barrier
            
            #pragma omp for schedule(static)
            for (i = 0; i < nclusters; i++) {
                int total_len = 0;
                for (int t = 0; t < num_threads; t++) {
                    total_len += thread_new_centers_len[t][i];
                }
                
                if (total_len > 0) {
                    for (j = 0; j < nfeatures; j++) {
                        float sum = 0.0;
                        for (int t = 0; t < num_threads; t++) {
                            sum += thread_new_centers[t][i * nfeatures + j];
                        }
                        clusters[i][j] = sum / total_len;
                    }
                }
                
                for (int t = 0; t < num_threads; t++) {
                    thread_new_centers_len[t][i] = 0;
                    for (j = 0; j < nfeatures; j++) {
                        thread_new_centers[t][i * nfeatures + j] = 0.0;
                    }
                }
            }
        }
#else
        for (i = 0; i < npoints; i++) {
            /* find the index of nestest cluster centers */
            index = find_nearest_point(feature[i], nfeatures, clusters,
                                       nclusters);
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

        /* replace old cluster centers with new_centers */
        for (i = 0; i < nclusters; i++) {
            for (j = 0; j < nfeatures; j++) {
                if (new_centers_len[i] > 0)
                    clusters[i][j] = new_centers[i][j] / new_centers_len[i];
                new_centers[i][j] = 0.0; /* set back to 0 */
            }
            new_centers_len[i] = 0; /* set back to 0 */
        }
#endif

    } while (delta > threshold && loop++ < 500);

#ifdef _OPENMP
    free(thread_new_centers_len[0]);
    free(thread_new_centers_len);
    free(thread_new_centers[0]);
    free(thread_new_centers);
#endif

    free(new_centers[0]);
    free(new_centers);
    free(new_centers_len);

    return clusters;
}
