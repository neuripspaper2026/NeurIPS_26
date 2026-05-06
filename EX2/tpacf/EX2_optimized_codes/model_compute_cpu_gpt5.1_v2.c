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
  /* Parallelize outer loop with OpenMP; use private temporaries
   * and a per-thread histogram reduced into data_bins at the end.
   */
  #pragma omp parallel
  {
    int i_local, j_local;
    long long *local_bins = (long long *)calloc((size_t)nbins + 2, sizeof(long long));
    if (local_bins == NULL) {
      /* Fallback to serial behavior if allocation fails */
      #pragma omp single
      {
        for (i = 0; i < (doSelf ? n1 - 1 : n1); i++) {
          const float xi = data1[i].x;
          const float yi = data1[i].y;
          const float zi = data1[i].z;

          for (j = (doSelf ? i + 1 : 0); j < n2; j++) {
            const float dot = xi * data2[j].x + yi * data2[j].y +
                              zi * data2[j].z;

            /* Binary search over bin boundaries */
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
      for (i_local = 0; i_local < (doSelf ? n1 - 1 : n1); i_local++) {
        const float xi = data1[i_local].x;
        const float yi = data1[i_local].y;
        const float zi = data1[i_local].z;

        const int j_start = doSelf ? (i_local + 1) : 0;
        for (j_local = j_start; j_local < n2; j_local++) {
          const float dot = xi * data2[j_local].x +
                            yi * data2[j_local].y +
                            zi * data2[j_local].z;

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

      /* Reduce local bins into global bins */
      #pragma omp critical
      {
        int k;
        for (k = 0; k < nbins + 2; ++k) {
          data_bins[k] += local_bins[k];
        }
      }

      free(local_bins);
    }
  }
#else
  /* Serial execution path */
  for (i = 0; i < (doSelf ? n1 - 1 : n1); i++) {
    const float xi = data1[i].x;
    const float yi = data1[i].y;
    const float zi = data1[i].z;

    for (j = (doSelf ? i + 1 : 0); j < n2; j++) {
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

