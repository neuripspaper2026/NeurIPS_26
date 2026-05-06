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
  int i;
  if (doSelf) {
    n2 = n1;
    data2 = data1;
  }

#ifdef _OPENMP
  /* Parallelize the outer loop; use a private temporary histogram per thread
     to avoid contention on data_bins, then reduce at the end. */
  long long *thread_bins = NULL;
  int nthreads = 1;

#pragma omp parallel
  {
    int tid_local = 0;
#ifdef _OPENMP
    tid_local = omp_get_thread_num();
#endif
#pragma omp single
    {
#ifdef _OPENMP
      nthreads = omp_get_num_threads();
#else
      nthreads = 1;
#endif
      thread_bins = (long long *)calloc((size_t)nthreads * (size_t)(nbins + 1), sizeof(long long));
    }

    /* Ensure allocation completed before use */
#pragma omp barrier

    long long *local_bins = thread_bins + (size_t)tid_local * (size_t)(nbins + 1);

    if (doSelf) {
#pragma omp for schedule(static)
      for (i = 0; i < n1 - 1; i++) {
        const float xi = data1[i].x;
        const float yi = data1[i].y;
        const float zi = data1[i].z;

        for (int j = i + 1; j < n2; j++) {
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
    } else {
#pragma omp for schedule(static)
      for (i = 0; i < n1; i++) {
        const float xi = data1[i].x;
        const float yi = data1[i].y;
        const float zi = data1[i].z;

        for (int j = 0; j < n2; j++) {
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
    }
  } /* end parallel region */

  /* Reduce per-thread histograms into the global data_bins */
  if (thread_bins != NULL) {
    for (int t = 0; t < nthreads; ++t) {
      long long *src = thread_bins + (size_t)t * (size_t)(nbins + 1);
      for (int b = 0; b <= nbins; ++b) {
        data_bins[b] += src[b];
      }
    }
    free(thread_bins);
  }

#else  /* no OpenMP: original serial implementation, lightly cleaned */

  if (doSelf) {
    for (i = 0; i < n1 - 1; i++) {
      const float xi = data1[i].x;
      const float yi = data1[i].y;
      const float zi = data1[i].z;
      
      for (int j = i + 1; j < n2; j++) {
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
      
      for (int j = 0; j < n2; j++) {
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

#endif /* _OPENMP */

  return 0;
}

