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

  const int end_i = doSelf ? (n1 - 1) : n1;

  for (int i = 0; i < end_i; ++i) {
    const float xi = data1[i].x;
    const float yi = data1[i].y;
    const float zi = data1[i].z;

    const int start_j = doSelf ? (i + 1) : 0;

    for (int j = start_j; j < n2; ++j) {
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
        ++data_bins[min];
      } else if (dot < binb[max]) {
        ++data_bins[max + 1];
      } else {
        ++data_bins[max];
      }
    }
  }

  return 0;
}

