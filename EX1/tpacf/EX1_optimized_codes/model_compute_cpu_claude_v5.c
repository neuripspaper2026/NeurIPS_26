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
  
  struct cartesian *d2 = data2;
  int limit1 = n1;
  int limit2 = n2;
  
  if (doSelf)
    {
      limit1 = n1 - 1;
      d2 = data1;
    }
  
  for (i = 0; i < limit1; i++)
    {
      const float xi = data1[i].x;
      const float yi = data1[i].y;
      const float zi = data1[i].z;
      
      int start_j = doSelf ? i + 1 : 0;
      
      for (j = start_j; j < limit2; j++)
        {
	  float dot = xi * d2[j].x + yi * d2[j].y + zi * d2[j].z;
	  
	  int min = 0;
	  int max = nbins;
	  
	  while (max > min + 1)
            {
	      int k = (min + max) >> 1;
	      if (dot >= binb[k]) 
		max = k;
	      else 
		min = k;
            }
	  
	  int indx;
	  if (dot >= binb[min]) 
	    {
	      indx = min;
	    }
	  else if (dot < binb[max]) 
	    { 
	      indx = max + 1;
	    }
	  else 
	    { 
	      indx = max;
	    }
	  
	  data_bins[indx] += 1;
        }
    }
  
  return 0;
}

