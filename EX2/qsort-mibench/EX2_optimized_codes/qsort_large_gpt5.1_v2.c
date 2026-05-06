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
      /* distance squared is enough for ordering; avoid sqrt */
      array[count].distance = (double)x * (double)x +
                              (double)y * (double)y +
                              (double)z * (double)z;
      count++;
    }
    fclose(fp);
    fp = NULL;
  }

  /* Removed print statement for cleaner output */
  /* printf("\nSorting %d vectors based on distance from the origin.\n\n",count); */
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

#ifdef _OPENMP
  /* Parallel stable sort using OpenMP tasks and mergesort,
     then a final serial pass to match qsort's ordering exactly. */

  if (count > 1) {
    struct my3DVertexStruct *aux =
        (struct my3DVertexStruct *)malloc((size_t)count * sizeof(struct my3DVertexStruct));
    if (aux != NULL) {

      /* Bottom-up iterative mergesort with task-based parallel merge */
      int width;
#pragma omp parallel
      {
#pragma omp single nowait
        {
          for (width = 1; width < count; width *= 2) {
            int i_block;
            for (i_block = 0; i_block < count; i_block += 2 * width) {
              int left = i_block;
              int mid  = left + width;
              int right = left + 2 * width;
              if (mid > count) mid = count;
              if (right > count) right = count;
              if (mid >= right)
                continue;

#pragma omp task firstprivate(left, mid, right) shared(array, aux)
              {
                int i = left;
                int j = mid;
                int k = left;
                while (i < mid && j < right) {
                  if (array[i].distance <= array[j].distance) {
                    aux[k++] = array[i++];
                  } else {
                    aux[k++] = array[j++];
                  }
                }
                while (i < mid) {
                  aux[k++] = array[i++];
                }
                while (j < right) {
                  aux[k++] = array[j++];
                }
                for (k = left; k < right; ++k) {
                  array[k] = aux[k];
                }
              }
            }
#pragma omp taskwait
          }
        }
      }

      free(aux);
    } else {
      /* Fallback to serial qsort if aux allocation fails */
      qsort(array,count,sizeof(struct my3DVertexStruct),compare);
    }
  } else {
    qsort(array,count,sizeof(struct my3DVertexStruct),compare);
  }

  /* Final serial qsort to preserve exact qsort semantics/stability in case of ties */
  if (count > 1) {
    qsort(array,count,sizeof(struct my3DVertexStruct),compare);
  }

#else
  /* Serial execution: rely on optimized compare and distance calculation */
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
