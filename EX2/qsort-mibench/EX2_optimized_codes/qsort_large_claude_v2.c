#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define UNLIMIT
#define MAXARRAY 31000000

struct my3DVertexStruct {
  int x, y, z;
  double distance;
};

int compare(const void *elem1, const void *elem2)
{
  double distance1, distance2;

  distance1 = (*((struct my3DVertexStruct *)elem1)).distance;
  distance2 = (*((struct my3DVertexStruct *)elem2)).distance;

  return (distance1 > distance2) ? 1 : ((distance1 == distance2) ? 0 : -1);
}

static void parallel_qsort(struct my3DVertexStruct *array, int left, int right, int depth) {
  if (left >= right) return;
  
  int i = left, j = right;
  struct my3DVertexStruct pivot = array[(left + right) / 2];
  struct my3DVertexStruct temp;
  
  while (i <= j) {
    while (compare(&array[i], &pivot) < 0) i++;
    while (compare(&array[j], &pivot) > 0) j--;
    if (i <= j) {
      temp = array[i];
      array[i] = array[j];
      array[j] = temp;
      i++;
      j--;
    }
  }
  
  if (depth > 0 && (right - left) > 10000) {
    #pragma omp task
    parallel_qsort(array, left, j, depth - 1);
    #pragma omp task
    parallel_qsort(array, i, right, depth - 1);
    #pragma omp taskwait
  } else {
    if (left < j) parallel_qsort(array, left, j, depth - 1);
    if (i < right) parallel_qsort(array, i, right, depth - 1);
  }
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
    
    while((fscanf(fp, "%d", &x) == 1) && (fscanf(fp, "%d", &y) == 1) && (fscanf(fp, "%d", &z) == 1) &&  (count < MAXARRAY)) {
	 array[count].x = x;
	 array[count].y = y;
	 array[count].z = z;
	 count++;
    }
    fclose(fp);
    fp = NULL;
  }
  
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for(i=0;i<count;i++) {
    int xi = array[i].x;
    int yi = array[i].y;
    int zi = array[i].z;
    array[i].distance = sqrt((double)(xi * xi + yi * yi + zi * zi));
  }
  
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);
  
  #ifdef _OPENMP
  #pragma omp parallel
  {
    #pragma omp single
    {
      parallel_qsort(array, 0, count - 1, 4);
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
