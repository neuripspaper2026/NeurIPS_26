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
  if (doSelf) {
    n2 = n1;
    data2 = data1;
  }

#ifdef _OPENMP
  /* Parallelize the outer loop i with OpenMP; use per-thread private
   * histograms to avoid false sharing and contention, then reduce. */
  {
    int i;

    /* Zero the global histogram once before entering parallel region. */
    for (int b = 0; b <= nbins; ++b) {
      data_bins[b] = 0;
    }

    #pragma omp parallel private(i)
    {
      /* Allocate and initialize a private histogram for each thread. */
      long long *local_bins = (long long *)calloc((size_t)(nbins + 1), sizeof(long long));
      if (!local_bins) {
        /* Fallback: no per-thread histogram; use direct atomic updates. */
        #pragma omp for schedule(static)
        for (i = 0; i < (doSelf ? n1 - 1 : n1); ++i) {
          const float xi = data1[i].x;
          const float yi = data1[i].y;
          const float zi = data1[i].z;

          int j_start = doSelf ? i + 1 : 0;
          for (int j = j_start; j < n2; ++j) {
            const float dot = xi * data2[j].x + yi * data2[j].y + zi * data2[j].z;

            int min = 0;
            int max = nbins;

            while (max > min + 1) {
              const int k = (min + max) >> 1;
              if (dot >= binb[k])
                max = k;
              else
                min = k;
            }

            int idx;
            if (dot >= binb[min]) {
              idx = min;
            } else if (dot < binb[max]) {
              idx = max + 1;
            } else {
              idx = max;
            }

            #pragma omp atomic
            data_bins[idx] += 1;
          }
        }
      } else {
        /* Histogram accumulation without atomics, using thread-private bins. */
        #pragma omp for schedule(static)
        for (i = 0; i < (doSelf ? n1 - 1 : n1); ++i) {
          const float xi = data1[i].x;
          const float yi = data1[i].y;
          const float zi = data1[i].z;

          int j_start = doSelf ? i + 1 : 0;
          for (int j = j_start; j < n2; ++j) {
            const float dot = xi * data2[j].x + yi * data2[j].y + zi * data2[j].z;

            int min = 0;
            int max = nbins;

            while (max > min + 1) {
              const int k = (min + max) >> 1;
              if (dot >= binb[k])
                max = k;
              else
                min = k;
            }

            int idx;
            if (dot >= binb[min]) {
              idx = min;
            } else if (dot < binb[max]) {
              idx = max + 1;
            } else {
              idx = max;
            }

            local_bins[idx] += 1;
          }
        }

        /* Reduction of per-thread histograms into the global one. */
        #pragma omp for schedule(static)
        for (int b = 0; b <= nbins; ++b) {
          long long sum = 0;
          sum += local_bins[b];
          #pragma omp atomic
          data_bins[b] += sum;
        }

        free(local_bins);
      }
    } /* end parallel */
  }
#else
  /* Serial fallback implementation. */
  int i;
  for (i = 0; i < (doSelf ? n1 - 1 : n1); i++) {
    const float xi = data1[i].x;
    const float yi = data1[i].y;
    const float zi = data1[i].z;

    int j_start = doSelf ? i + 1 : 0;
    for (int j = j_start; j < n2; j++) {
      const float dot = xi * data2[j].x + yi * data2[j].y + zi * data2[j].z;

      int min = 0;
      int max = nbins;

      while (max > min + 1) {
        const int k = (min + max) >> 1;
        if (dot >= binb[k])
          max = k;
        else
          min = k;
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

