#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define UNLIMIT
#define MAXARRAY 31000000 /* Increased to support larger datasets (up to 30M vertices) */

struct my3DVertexStruct {
  int x, y, z;
  double distance;
};

int compare(const void *elem1, const void *elem2)
{
  /* D = [(x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2]^(1/2) */
  /* sort based on distances from the origin... */

  const struct my3DVertexStruct *a = (const struct my3DVertexStruct *)elem1;
  const struct my3DVertexStruct *b = (const struct my3DVertexStruct *)elem2;

  if (a->distance < b->distance) return -1;
  if (a->distance > b->distance) return 1;
  return 0;
}

int
main(int argc, char *argv[]) {
  struct my3DVertexStruct *array = NULL;
  FILE *fp = NULL;
  int i,count=0;
  int x, y, z;
  int ret = 0;
  struct timespec main_start, main_end;
  struct timespec kernel_start, kernel_end;
  int kernel_measured = 0;
  FILE *timing_file = stderr;
  const char *timing_path = getenv("TIMING_LOG_FILE");

  clock_gettime(CLOCK_MONOTONIC, &main_start);
  if (timing_path && timing_path[0] != '\0') {
    FILE *tmp = fopen(timing_path, "w");
    if (tmp) {
      timing_file = tmp;
    }
  }

  /* Dynamically allocate array to avoid stack overflow */
  array = (struct my3DVertexStruct *)malloc((size_t)MAXARRAY * sizeof(struct my3DVertexStruct));
  if (array == NULL) {
    fprintf(stderr, "Error: Failed to allocate memory for %d vertices\n", MAXARRAY);
    ret = -1;
    goto cleanup;
  }

  if (argc<2) {
    fprintf(stderr,"Usage: qsort_large <file>\n");
    ret = -1;
    goto cleanup;
  }
  else {
    fp = fopen(argv[1],"r");
    if (fp == NULL) {
      fprintf(stderr, "Error: Cannot open file %s\n", argv[1]);
      ret = -1;
      goto cleanup;
    }

    while ((count < MAXARRAY) &&
           (fscanf(fp, "%d %d %d", &x, &y, &z) == 3)) {
      array[count].x = x;
      array[count].y = y;
      array[count].z = z;
      /* use squared distance for ordering to avoid sqrt, pow */
      double dx = (double)x;
      double dy = (double)y;
      double dz = (double)z;
      array[count].distance = dx*dx + dy*dy + dz*dz;
      count++;
    }
    fclose(fp);
    fp = NULL;
  }
  /* Removed print statement for cleaner output */
  /* printf("\nSorting %d vectors based on distance from the origin.\n\n",count); */

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

#ifdef _OPENMP
  if (count > 100000) {
    /* Parallel sort for large datasets: sort chunks in parallel, then merge serially */
    int num_threads = 1;
#pragma omp parallel
    {
#pragma omp single
      num_threads = omp_get_num_threads();
    }

    if (num_threads > 1) {
      int t;
#pragma omp parallel private(t)
      {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
        size_t chunk = (size_t)count / (size_t)nt;
        size_t start = (size_t)tid * chunk;
        size_t end = (tid == nt - 1) ? (size_t)count : start + chunk;
        if (start < end) {
          qsort(array + start, end - start, sizeof(struct my3DVertexStruct), compare);
        }
      }

      /* k-way merge performed serially */
      int *idx = (int *)malloc((size_t)num_threads * sizeof(int));
      if (idx != NULL) {
        size_t *start_arr = (size_t *)malloc((size_t)num_threads * sizeof(size_t));
        size_t *end_arr   = (size_t *)malloc((size_t)num_threads * sizeof(size_t));
        struct my3DVertexStruct *tmp =
            (struct my3DVertexStruct *)malloc((size_t)count * sizeof(struct my3DVertexStruct));

        if (start_arr != NULL && end_arr != NULL && tmp != NULL) {
          for (t = 0; t < num_threads; ++t) {
            size_t chunk = (size_t)count / (size_t)num_threads;
            start_arr[t] = (size_t)t * chunk;
            end_arr[t]   = (t == num_threads - 1) ? (size_t)count : start_arr[t] + chunk;
            idx[t] = (int)start_arr[t];
          }

          size_t out = 0;
          while (out < (size_t)count) {
            int best = -1;
            for (t = 0; t < num_threads; ++t) {
              if ((size_t)idx[t] < end_arr[t]) {
                if (best == -1 ||
                    array[idx[t]].distance < array[idx[best]].distance) {
                  best = t;
                }
              }
            }
            if (best == -1) break;
            tmp[out++] = array[idx[best]++];
          }

          if (out == (size_t)count) {
            for (size_t k = 0; k < (size_t)count; ++k) {
              array[k] = tmp[k];
            }
          }
        }

        if (tmp) free(tmp);
        if (start_arr) free(start_arr);
        if (end_arr) free(end_arr);
        free(idx);
      }
    } else {
      qsort(array,count,sizeof(struct my3DVertexStruct),compare);
    }
  } else {
    qsort(array,count,sizeof(struct my3DVertexStruct),compare);
  }
#else
  qsort(array,count,sizeof(struct my3DVertexStruct),compare);
#endif

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  kernel_measured = 1;

  for(i=0;i<count;i++)
    printf("%d %d %d\n", array[i].x, array[i].y, array[i].z);

  ret = 0;

cleanup:
  if (fp) {
    fclose(fp);
  }
  if (array) {
    free(array);
  }
  clock_gettime(CLOCK_MONOTONIC, &main_end);
  double kernel_time = 0.0;
  if (kernel_measured) {
    kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
  }
  double main_time = (main_end.tv_sec - main_start.tv_sec) +
                     (main_end.tv_nsec - main_start.tv_nsec) / 1e9;
  fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
  fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);
  fflush(timing_file);
  if (timing_file != stderr) {
    fclose(timing_file);
  }
  return ret;
}
