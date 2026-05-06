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

  /* Preload boundaries into local array to improve cache locality
   * and avoid repeated pointer dereferences inside inner loops. */
  float binb_local[NUM_BINS + 1];
  const int max_bins = nbins;
  for (int b = 0; b <= max_bins; ++b) {
    binb_local[b] = binb[b];
  }

  const int outerLimit = doSelf ? (n1 - 1) : n1;

  for (int i = 0; i < outerLimit; ++i) {
    const float xi = data1[i].x;
    const float yi = data1[i].y;
    const float zi = data1[i].z;

    const int innerStart = doSelf ? (i + 1) : 0;

    for (int j = innerStart; j < n2; ++j) {
      const float dot = xi * data2[j].x + yi * data2[j].y +
                        zi * data2[j].z;

      int min = 0;
      int max = max_bins;

      /* Binary search on preloaded bin boundaries */
      while (max > min + 1) {
        const int mid = (min + max) >> 1;
        if (dot >= binb_local[mid])
          max = mid;
        else
          min = mid;
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

  return 0;
}

