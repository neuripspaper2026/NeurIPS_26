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

  const struct my3DVertexStruct *v1 = (const struct my3DVertexStruct *)elem1;
  const struct my3DVertexStruct *v2 = (const struct my3DVertexStruct *)elem2;

  if (v1->distance > v2->distance)
    return 1;
  else if (v1->distance < v2->distance)
    return -1;
  else
    return 0;
}

static inline double squared_distance_int(int x, int y, int z)
{
  /* Use double for accumulation but avoid pow/sqrt for performance */
  double dx = (double)x;
  double dy = (double)y;
  double dz = (double)z;
  return dx * dx + dy * dy + dz * dz;
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

    /* Read all points first, then compute distances in a parallelizable loop */
    while ((fscanf(fp, "%d", &x) == 1) && (fscanf(fp, "%d", &y) == 1) &&
           (fscanf(fp, "%d", &z) == 1) && (count < MAXARRAY)) {
      array[count].x = x;
      array[count].y = y;
      array[count].z = z;
      count++;
    }
    fclose(fp);
    fp = NULL;

#ifdef _OPENMP
    /* Parallel computation of distances with OpenMP */
    #pragma omp parallel for schedule(static)
    for (i = 0; i < count; i++) {
      int lx = array[i].x;
      int ly = array[i].y;
      int lz = array[i].z;
      array[i].distance = squared_distance_int(lx, ly, lz);
    }
#else
    for (i = 0; i < count; i++) {
      int lx = array[i].x;
      int ly = array[i].y;
      int lz = array[i].z;
      array[i].distance = squared_distance_int(lx, ly, lz);
    }
#endif
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);
  qsort(array,count,sizeof(struct my3DVertexStruct),compare);
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
