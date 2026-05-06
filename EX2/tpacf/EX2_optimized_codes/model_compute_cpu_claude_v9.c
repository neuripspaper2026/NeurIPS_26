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
  
  const int outer_limit = (doSelf) ? n1-1 : n1;
  
#ifdef _OPENMP
  // Thread-local histogram bins to avoid false sharing and atomic overhead
  int num_threads = omp_get_max_threads();
  long long *local_bins = (long long *)calloc(num_threads * (nbins + 2), sizeof(long long));
  
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    long long *my_bins = local_bins + tid * (nbins + 2);
    
    #pragma omp for schedule(dynamic, 64) nowait
    for (i = 0; i < outer_limit; i++)
      {
        const float xi = data1[i].x;
        const float yi = data1[i].y;
        const float zi = data1[i].z;
        
        const int inner_start = (doSelf) ? i+1 : 0;
        
        for (j = inner_start; j < n2; j++)
          {
            const float dot = xi * data2[j].x + yi * data2[j].y + zi * data2[j].z;
            
            // Binary search
            int min = 0;
            int max = nbins;
            
            while (max > min+1)
              {
                int k = (min + max) >> 1;
                if (dot >= binb[k]) 
                  max = k;
                else 
                  min = k;
              }
            
            if (dot >= binb[min]) 
              {
                my_bins[min] += 1;
              }
            else if (dot < binb[max]) 
              { 
                my_bins[max+1] += 1;
              }
            else 
              { 
                my_bins[max] += 1;
              }
          }
      }
  }
  
  // Reduce thread-local bins into global data_bins
  for (int t = 0; t < num_threads; t++)
    {
      long long *my_bins = local_bins + t * (nbins + 2);
      for (int b = 0; b < nbins + 2; b++)
        {
          data_bins[b] += my_bins[b];
        }
    }
  
  free(local_bins);
#else
  // Serial version
  for (i = 0; i < outer_limit; i++)
    {
      const float xi = data1[i].x;
      const float yi = data1[i].y;
      const float zi = data1[i].z;
      
      const int inner_start = (doSelf) ? i+1 : 0;
      
      for (j = inner_start; j < n2; j++)
        {
          const float dot = xi * data2[j].x + yi * data2[j].y + zi * data2[j].z;
          
          // Binary search
          int min = 0;
          int max = nbins;
          
          while (max > min+1)
            {
              int k = (min + max) >> 1;
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
              data_bins[max+1] += 1;
            }
          else 
            { 
              data_bins[max] += 1;
            }
        }
    }
#endif
  
  return 0;
}

