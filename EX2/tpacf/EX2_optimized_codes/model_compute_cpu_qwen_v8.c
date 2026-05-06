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
  #pragma omp parallel for private(j) reduction(+:data_bins[0:nbins+1])
#endif
  for (i = 0; i < ((doSelf) ? n1-1 : n1); i++)
    {
      const float xi = data1[i].x;
      const float yi = data1[i].y;
      const float zi = data1[i].z;
      
      for (j = ((doSelf) ? i+1 : 0); j < n2; j++)
        {
	  float dot = xi * data2[j].x + yi * data2[j].y + zi * data2[j].z;
	  
	  // run binary search
	  int min = 0;
	  int max = nbins;
	  int k;
	  
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
	      data_bins[min] += 1;
	    }
	  else if (dot < binb[max]) 
	    { 
	      data_bins[max+1] += 1;
	    }
	  else 
	    { 
	      data_bins[max] += 1;
	    }
        }
    }
  
  return 0;
}

