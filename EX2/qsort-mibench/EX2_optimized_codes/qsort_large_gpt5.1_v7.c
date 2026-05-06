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

  double distance1, distance2;

  distance1 = ((const struct my3DVertexStruct *)elem1)->distance;
  distance2 = ((const struct my3DVertexStruct *)elem2)->distance;

  return (distance1 > distance2) ? 1 : ((distance1 == distance2) ? 0 : -1);
}

static inline double compute_distance(int x, int y, int z) {
  return sqrt((double)x * (double)x +
              (double)y * (double)y +
              (double)z * (double)z);
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
  array = (struct my3DVertexStruct *)malloc(MAXARRAY * sizeof(struct my3DVertexStruct));
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

    while ((fscanf(fp, "%d %d %d", &x, &y, &z) == 3) && (count < MAXARRAY)) {
      array[count].x = x;
      array[count].y = y;
      array[count].z = z;
      array[count].distance = compute_distance(x, y, z);
      count++;
    }
    fclose(fp);
    fp = NULL;
  }
  /* Removed print statement for cleaner output */
  /* printf("\nSorting %d vectors based on distance from the origin.\n\n",count); */
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

#ifdef _OPENMP
  if (count > 1) {
    int num_threads = 1;
    #pragma omp parallel
    {
      #pragma omp single
      num_threads = omp_get_num_threads();
    }

    if (num_threads > 1) {
      int t;
      int *starts = (int *)malloc((size_t)num_threads * sizeof(int));
      int *ends   = (int *)malloc((size_t)num_threads * sizeof(int));
      if (starts && ends) {
        int base = count / num_threads;
        int rem  = count % num_threads;
        int idx = 0;
        for (t = 0; t < num_threads; ++t) {
          int len = base + (t < rem ? 1 : 0);
          starts[t] = idx;
          ends[t]   = idx + len;
          idx      += len;
        }

        #pragma omp parallel
        {
          int tid = omp_get_thread_num();
          int s = starts[tid];
          int e = ends[tid];
          if (e - s > 1) {
            qsort(&array[s], (size_t)(e - s), sizeof(struct my3DVertexStruct), compare);
          }
        }

        int current_threads = num_threads;
        int step;
        for (step = 1; step < current_threads; step <<= 1) {
          int new_threads = 0;

          #pragma omp parallel
          {
            int tid = omp_get_thread_num();
            #pragma omp for schedule(static) reduction(+:new_threads)
            for (t = 0; t + step < current_threads; t += (step << 1)) {
              int left  = starts[t];
              int mid   = ends[t];
              int right = ends[t + step];

              int n1 = mid - left;
              int n2 = right - mid;

              struct my3DVertexStruct *temp =
                  (struct my3DVertexStruct *)malloc((size_t)(n1 + n2) * sizeof(struct my3DVertexStruct));
              if (!temp) {
                continue;
              }

              int i1 = 0, i2 = 0, k = 0;
              while (i1 < n1 && i2 < n2) {
                if (array[left + i1].distance <= array[mid + i2].distance) {
                  temp[k++] = array[left + i1++];
                } else {
                  temp[k++] = array[mid + i2++];
                }
              }
              while (i1 < n1) {
                temp[k++] = array[left + i1++];
              }
              while (i2 < n2) {
                temp[k++] = array[mid + i2++];
              }

              for (k = 0; k < n1 + n2; ++k) {
                array[left + k] = temp[k];
              }

              free(temp);

              ends[t] = right;
              new_threads++;
            }
          }

          if (new_threads == 0) {
            break;
          }
          for (t = 0, idx = 0; t < current_threads; t += (step << 1), ++idx) {
            starts[idx] = starts[t];
            ends[idx]   = ends[t];
          }
          current_threads = idx;
        }
      }

      if (starts) free(starts);
      if (ends)   free(ends);
    } else {
      qsort(array, (size_t)count, sizeof(struct my3DVertexStruct), compare);
    }
  }
  else {
    if (count == 1) {
      /* nothing to sort */
    }
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
