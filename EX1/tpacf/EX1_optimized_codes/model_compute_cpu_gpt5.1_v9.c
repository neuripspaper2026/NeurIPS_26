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

  if (doSelf) {
    n2 = n1;
    data2 = data1;
  }

  /* Preload bin boundaries into a local array to improve cache locality
     and avoid repeated indirections through binb. Assumes nbins is
     reasonably small (NUM_BINS). */
  float local_binb[NUM_BINS + 1];
  int nbins_local = nbins;
  for (i = 0; i <= nbins_local; ++i) {
    local_binb[i] = binb[i];
  }

  for (i = 0; i < (doSelf ? n1 - 1 : n1); i++) {
    const float xi = data1[i].x;
    const float yi = data1[i].y;
    const float zi = data1[i].z;

    /* Loop over j: when doSelf is true, we only consider j > i */
    const int j_start = doSelf ? (i + 1) : 0;
    for (j = j_start; j < n2; j++) {
      const float dot =
          xi * data2[j].x +
          yi * data2[j].y +
          zi * data2[j].z;

      /* Binary search over local_binb */
      int min = 0;
      int max = nbins_local;

      while (max > min + 1) {
        const int mid = (min + max) >> 1;
        if (dot >= local_binb[mid])
          max = mid;
        else
          min = mid;
      }

      if (dot >= local_binb[min]) {
        ++data_bins[min];
      } else if (dot < local_binb[max]) {
        ++data_bins[max + 1];
      } else {
        ++data_bins[max];
      }
    }
  }

  return 0;
}

