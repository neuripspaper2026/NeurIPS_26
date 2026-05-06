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
  if (doSelf)
    {
      n2 = n1;
      data2 = data1;
    }

#ifdef _OPENMP
  const int nthreads = omp_get_max_threads();
  long long (*thread_bins)[NUM_BINS+2] = calloc(nthreads, sizeof(long long[NUM_BINS+2]));
#else
  long long * const thread_bins[1] = { data_bins };
#endif

#pragma omp parallel for schedule(dynamic, 16)
  for (i = 0; i < ((doSelf) ? n1-1 : n1); i++)
    {
      const register float xi = data1[i].x;
      const register float yi = data1[i].y;
      const register float zi = data1[i].z;

#ifdef _OPENMP
      long long * const data_bins = thread_bins[omp_get_thread_num()];
#endif

      for (j = ((doSelf) ? i+1 : 0); j < n2; j++)
        {
	  register float dot = xi * data2[j].x + yi * data2[j].y + 
	    zi * data2[j].z;
	  
	  // run binary search
	  register int min = 0;
	  register int max = nbins;
	  register int k;
	  
	  while (max > min+1)
            {
	      k = (min + max) / 2;
	      if (dot >= binb[k]) 
		max = k;
	      else 
		min = k;
            };
	  
	  if (dot >= binb[min]) 
	    {
	      data_bins[min] += 1; /*k = min;*/ 
	    }
	  else if (dot < binb[max]) 
	    { 
	      data_bins[max+1] += 1; /*k = max+1;*/ 
	    }
	  else 
	    { 
	      data_bins[max] += 1; /*k = max;*/ 
	    }
        }
    }

#ifdef _OPENMP
  for (int t = 0; t < nthreads; t++) {
    for (int b = 0; b < nbins+2; b++) {
      data_bins[b] += thread_bins[t][b];
    }
  }
  free(thread_bins);
#endif
  
  return 0;
}

