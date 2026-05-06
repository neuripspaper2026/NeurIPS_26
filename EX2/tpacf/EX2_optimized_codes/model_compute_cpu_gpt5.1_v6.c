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

  const int i_end = (doSelf ? (n1 - 1) : n1);

#ifdef _OPENMP
  /* Parallelize outer loop; use per-thread private histograms to
   * avoid contention, then reduce into data_bins. */
  #pragma omp parallel
  {
    long long *local_bins = (long long *)calloc((size_t)nbins, sizeof(long long));
    if (!local_bins) {
      /* Fallback to single-threaded behavior if allocation fails */
      #pragma omp single
      {
        int i, j;
        for (i = 0; i < i_end; i++) {
          const float xi = data1[i].x;
          const float yi = data1[i].y;
          const float zi = data1[i].z;

          const int j_start = doSelf ? (i + 1) : 0;
          for (j = j_start; j < n2; j++) {
            const float dot = xi * data2[j].x + yi * data2[j].y +
                              zi * data2[j].z;

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
      }
    } else {
      #pragma omp for schedule(static)
      for (int i = 0; i < i_end; i++) {
        const float xi = data1[i].x;
        const float yi = data1[i].y;
        const float zi = data1[i].z;

        const int j_start = doSelf ? (i + 1) : 0;
        for (int j = j_start; j < n2; j++) {
          const float dot = xi * data2[j].x + yi * data2[j].y +
                            zi * data2[j].z;

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
            local_bins[min] += 1;
          } else if (dot < binb[max]) {
            local_bins[max + 1] += 1;
          } else {
            local_bins[max] += 1;
          }
        }
      }

      /* Reduce local_bins into global data_bins */
      #pragma omp critical
      {
        for (int b = 0; b < nbins; b++) {
          data_bins[b] += local_bins[b];
        }
      }

      free(local_bins);
    }
  }
#else
  /* Serial version */
  for (int i = 0; i < i_end; i++) {
    const float xi = data1[i].x;
    const float yi = data1[i].y;
    const float zi = data1[i].z;

    const int j_start = doSelf ? (i + 1) : 0;
    for (int j = j_start; j < n2; j++) {
      const float dot = xi * data2[j].x + yi * data2[j].y +
                        zi * data2[j].z;

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

