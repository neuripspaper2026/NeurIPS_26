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
  /* Parallelize outer loop; use private temporaries and thread-local bins */
  #pragma omp parallel
  {
    int i_local, j_local;
    float xi, yi, zi;

    /* each thread gets its own accumulator array to avoid false sharing */
    long long *local_bins = (long long *)calloc((size_t)nbins + 2, sizeof(long long));

    #pragma omp for schedule(static)
    for (i_local = 0; i_local < (doSelf ? n1 - 1 : n1); i_local++) {
      xi = data1[i_local].x;
      yi = data1[i_local].y;
      zi = data1[i_local].z;

      const int j_start = doSelf ? i_local + 1 : 0;

      for (j_local = j_start; j_local < n2; j_local++) {
        float dot = xi * data2[j_local].x +
                    yi * data2[j_local].y +
                    zi * data2[j_local].z;

        /* Binary search over monotonically ordered bin boundaries */
        int min = 0;
        int max = nbins;

        while (max > min + 1) {
          int k_mid = (min + max) >> 1;
          if (dot >= binb[k_mid])
            max = k_mid;
          else
            min = k_mid;
        }

        if (dot >= binb[min]) {
          local_bins[min] += 1;
        }
        else if (dot < binb[max]) {
          local_bins[max + 1] += 1;
        }
        else {
          local_bins[max] += 1;
        }
      }
    }

    /* Reduction of local_bins into global data_bins */
    #pragma omp critical
    {
      for (int k = 0; k < nbins + 2; ++k) {
        data_bins[k] += local_bins[k];
      }
    }

    free(local_bins);
  }

#else /* !_OPENMP */

  for (i = 0; i < (doSelf ? n1 - 1 : n1); i++) {
    const float xi = data1[i].x;
    const float yi = data1[i].y;
    const float zi = data1[i].z;

    const int j_start = doSelf ? i + 1 : 0;

    for (j = j_start; j < n2; j++) {
      float dot = xi * data2[j].x +
                  yi * data2[j].y +
                  zi * data2[j].z;

      int min = 0;
      int max = nbins;

      while (max > min + 1) {
        int k_mid = (min + max) >> 1;
        if (dot >= binb[k_mid])
          max = k_mid;
        else
          min = k_mid;
      }

      if (dot >= binb[min]) {
        data_bins[min] += 1;
      }
      else if (dot < binb[max]) {
        data_bins[max + 1] += 1;
      }
      else {
        data_bins[max] += 1;
      }
    }
  }

#endif /* _OPENMP */

  return 0;
}

