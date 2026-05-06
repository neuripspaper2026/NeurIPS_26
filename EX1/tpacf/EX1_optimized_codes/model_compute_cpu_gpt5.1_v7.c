#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <stdio.h> 

#include "model.h"

int doCompute(struct cartesian *data1, int n1, struct cartesian *data2, 
	      int n2, int doSelf, long long *data_bins, 
	      int nbins, float *binb)
{
  if (doSelf) {
    n2 = n1;
    data2 = data1;
  }

  /* Preload first and last bin boundaries to avoid repeated memory loads */
  const float first_bin = binb[0];
  const float last_bin  = binb[nbins];

  const int i_end = doSelf ? (n1 - 1) : n1;

  for (int i = 0; i < i_end; i++) {
    const float xi = data1[i].x;
    const float yi = data1[i].y;
    const float zi = data1[i].z;

    const int j_start = doSelf ? (i + 1) : 0;

    for (int j = j_start; j < n2; j++) {
      const float dot = xi * data2[j].x + yi * data2[j].y + zi * data2[j].z;

      /* Fast reject for values clearly outside the bin range */
      if (dot >= first_bin) {
        data_bins[0] += 1;
        continue;
      }
      if (dot < last_bin) {
        data_bins[nbins + 1] += 1;
        continue;
      }

      /* Binary search within valid range */
      int min = 0;
      int max = nbins;

      while (max > min + 1) {
        const int k = (min + max) >> 1;
        if (dot >= binb[k])
          max = k;
        else
          min = k;
      }

      /* Branchless index selection using known ordering:
       * binb[min] >= dot >= binb[max] or dot < binb[max]
       */
      int idx;
      if (dot >= binb[min]) {
        idx = min;
      } else if (dot < binb[max]) {
        idx = max + 1;
      } else {
        idx = max;
      }
      data_bins[idx] += 1;
    }
  }

  return 0;
}

