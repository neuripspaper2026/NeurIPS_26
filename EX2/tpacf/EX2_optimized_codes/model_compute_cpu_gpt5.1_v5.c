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

  /* Preload bin boundaries into a local array for faster access.
     This helps both serial and parallel execution by improving cache locality. */
  int nbins_local = nbins;
  float *binb_local = binb;

#ifdef _OPENMP
  /* Parallelize outer loop over i. Use static scheduling for better balance
     given regular loop bounds. Each thread accumulates into its own
     private histogram to avoid false sharing/contention, then reduces
     into the global data_bins at the end. */
  #pragma omp parallel
  {
    int i, j;
    long long *local_bins = (long long *)calloc((size_t)(nbins_local + 2), sizeof(long long));

    #pragma omp for schedule(static)
    for (i = 0; i < (doSelf ? n1 - 1 : n1); i++) {
      const float xi = data1[i].x;
      const float yi = data1[i].y;
      const float zi = data1[i].z;

      const int j_start = doSelf ? i + 1 : 0;
      for (j = j_start; j < n2; j++) {
        const float dot = xi * data2[j].x + yi * data2[j].y +
                          zi * data2[j].z;

        /* Binary search over dot against bin boundaries */
        int min = 0;
        int max = nbins_local;

        while (max > min + 1) {
          const int k = (min + max) >> 1;
          if (dot >= binb_local[k])
            max = k;
          else
            min = k;
        }

        if (dot >= binb_local[min]) {
          local_bins[min] += 1;
        } else if (dot < binb_local[max]) {
          local_bins[max + 1] += 1;
        } else {
          local_bins[max] += 1;
        }
      }
    }

    /* Reduction into global data_bins */
    #pragma omp for schedule(static)
    for (int k = 0; k < nbins_local + 2; k++) {
      data_bins[k] += local_bins[k];
    }

    free(local_bins);
  }
#else
  /* Serial version with minor micro-optimizations */
  int i, j;
  for (i = 0; i < (doSelf ? n1 - 1 : n1); i++) {
    const float xi = data1[i].x;
    const float yi = data1[i].y;
    const float zi = data1[i].z;

    const int j_start = doSelf ? i + 1 : 0;
    for (j = j_start; j < n2; j++) {
      const float dot = xi * data2[j].x + yi * data2[j].y +
                        zi * data2[j].z;

      int min = 0;
      int max = nbins_local;

      while (max > min + 1) {
        const int k = (min + max) >> 1;
        if (dot >= binb_local[k])
          max = k;
        else
          min = k;
      }

      if (dot >= binb_local[min]) {
        data_bins[min] += 1;
      } else if (dot < binb_local[max]) {
        data_bins[max + 1] += 1;
      } else {
        data_bins[max] += 1;
      }
    }
  }
#endif

  return 0;
}

