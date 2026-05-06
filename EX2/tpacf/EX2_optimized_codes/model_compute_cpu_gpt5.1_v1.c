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
  if (doSelf) {
    /* Self-correlation: only upper triangle (i < j) is processed.
       Parallelize with OpenMP and use a reduction over the histogram. */
#pragma omp parallel
    {
      /* Per-thread private histogram to avoid contention */
      long long *private_bins = (long long *)calloc((size_t)nbins + 2U, sizeof(long long));
      int i, j;

#pragma omp for schedule(static)
      for (i = 0; i < n1 - 1; i++) {
        const float xi = data1[i].x;
        const float yi = data1[i].y;
        const float zi = data1[i].z;

        for (j = i + 1; j < n2; j++) {
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
            private_bins[min] += 1;
          } else if (dot < binb[max]) {
            private_bins[max + 1] += 1;
          } else {
            private_bins[max] += 1;
          }
        }
      }

      /* Combine private histograms into the global one */
#pragma omp for schedule(static)
      for (i = 0; i < nbins + 2; i++) {
        long long sum = 0;
#pragma omp atomic read
        sum = data_bins[i];
        sum += private_bins[i];
#pragma omp atomic write
        data_bins[i] = sum;
      }

      free(private_bins);
    }
  } else {
    /* Cross-correlation: full rectangle (i over data1, j over data2). */
#pragma omp parallel
    {
      long long *private_bins = (long long *)calloc((size_t)nbins + 2U, sizeof(long long));
      int i, j;

#pragma omp for schedule(static)
      for (i = 0; i < n1; i++) {
        const float xi = data1[i].x;
        const float yi = data1[i].y;
        const float zi = data1[i].z;

        for (j = 0; j < n2; j++) {
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
            private_bins[min] += 1;
          } else if (dot < binb[max]) {
            private_bins[max + 1] += 1;
          } else {
            private_bins[max] += 1;
          }
        }
      }

#pragma omp for schedule(static)
      for (i = 0; i < nbins + 2; i++) {
        long long sum = 0;
#pragma omp atomic read
        sum = data_bins[i];
        sum += private_bins[i];
#pragma omp atomic write
        data_bins[i] = sum;
      }

      free(private_bins);
    }
  }
#else
  /* Serial version (no OpenMP) */
  int i, j;
  if (doSelf) {
    for (i = 0; i < n1 - 1; i++) {
      const float xi = data1[i].x;
      const float yi = data1[i].y;
      const float zi = data1[i].z;

      for (j = i + 1; j < n2; j++) {
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
  } else {
    for (i = 0; i < n1; i++) {
      const float xi = data1[i].x;
      const float yi = data1[i].y;
      const float zi = data1[i].z;

      for (j = 0; j < n2; j++) {
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
#endif

  return 0;
}

