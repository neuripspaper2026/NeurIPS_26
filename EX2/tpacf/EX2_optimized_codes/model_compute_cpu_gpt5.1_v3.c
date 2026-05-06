#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "model.h"

int doCompute(struct cartesian *data1, int n1, struct cartesian *data2,
              int n2, int doSelf, long long *data_bins,
              int nbins, float *binb)
{
  int i, j;
  if (doSelf) {
    n2 = n1;
    data2 = data1;
  }

#ifdef _OPENMP
  /* Parallelize outer loop over i. Each thread keeps a private histogram
     to avoid false sharing and atomic increments; these are reduced at end. */
  long long *thread_hist = NULL;
  int max_threads = 1;

  #pragma omp parallel
  {
    int tid = 0;
    int nthreads = 1;

    #pragma omp single
    {
      max_threads = omp_get_num_threads();
      thread_hist = (long long *)calloc((size_t)max_threads * (size_t)nbins,
                                        sizeof(long long));
    }

    tid = omp_get_thread_num();
    nthreads = max_threads;

    long long *local_bins = thread_hist + (size_t)tid * (size_t)nbins;

    #pragma omp for schedule(static) nowait
    for (i = 0; i < (doSelf ? n1 - 1 : n1); i++) {
      const float xi = data1[i].x;
      const float yi = data1[i].y;
      const float zi = data1[i].z;

      const int j_begin = doSelf ? i + 1 : 0;

      for (j = j_begin; j < n2; j++) {
        const float dot = xi * data2[j].x + yi * data2[j].y +
                          zi * data2[j].z;

        int min = 0;
        int max = nbins;

        while (max > min + 1) {
          const int mid = (min + max) >> 1;
          if (dot >= binb[mid])
            max = mid;
          else
            min = mid;
        }

        if (dot >= binb[min]) {
          local_bins[min] += 1;
        } else if (dot < binb[max]) {
          local_bins[max + 1] += 1;
        } else {
          local_bins[max] += 1;
        }
      }
    }
  }

  /* Reduce per-thread histograms into global data_bins */
  if (thread_hist != NULL) {
    for (int t = 0; t < max_threads; ++t) {
      const long long *src = thread_hist + (size_t)t * (size_t)nbins;
      for (int b = 0; b < nbins; ++b) {
        data_bins[b] += src[b];
      }
    }
    free(thread_hist);
  }
#else
  for (i = 0; i < (doSelf ? n1 - 1 : n1); i++) {
    const float xi = data1[i].x;
    const float yi = data1[i].y;
    const float zi = data1[i].z;

    const int j_begin = doSelf ? i + 1 : 0;

    for (j = j_begin; j < n2; j++) {
      const float dot = xi * data2[j].x + yi * data2[j].y +
                        zi * data2[j].z;

      int min = 0;
      int max = nbins;

      while (max > min + 1) {
        const int mid = (min + max) >> 1;
        if (dot >= binb[mid])
          max = mid;
        else
          min = mid;
      }

      if (dot >= binb[min]) {
        data_bins[min] += 1;
      } else if (dot < binb[max]) {
        data_bins[max + 1] += 1;
      } else {
        data_bins[max] += 1;
      }
    }
  }
#endif

  return 0;
}

