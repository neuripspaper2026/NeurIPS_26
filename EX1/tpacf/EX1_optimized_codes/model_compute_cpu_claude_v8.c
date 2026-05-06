#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <stdio.h> 

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
  
  const int loop_limit = doSelf ? n1 - 1 : n1;
  
  for (i = 0; i < loop_limit; i++)
    {
      const float xi = data1[i].x;
      const float yi = data1[i].y;
      const float zi = data1[i].z;
      
      const int j_start = doSelf ? i + 1 : 0;
      
      for (j = j_start; j < n2; j++)
        {
	  const float dot = xi * data2[j].x + yi * data2[j].y + zi * data2[j].z;
	  
	  int min = 0;
	  int max = nbins;
	  
	  while (max > min + 1)
            {
	      const int k = (min + max) >> 1;
	      if (dot >= binb[k]) 
		max = k;
	      else 
		min = k;
            }
	  
	  if (dot >= binb[min]) 
	    {
	      data_bins[min] += 1;
	    }
	  else if (dot < binb[max]) 
	    { 
	      data_bins[max + 1] += 1;
	    }
	  else 
	    { 
	      data_bins[max] += 1;
	    }
        }
    }
  
  return 0;
}

