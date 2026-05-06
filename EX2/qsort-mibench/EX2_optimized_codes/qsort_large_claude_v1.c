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
  
  int threshold = 10000;
  if (right - left < threshold || depth > 3) {
    qsort(array + left, right - left + 1, sizeof(struct my3DVertexStruct), compare);
    return;
  }
  
  int pivot_idx = left + (right - left) / 2;
  double pivot = array[pivot_idx].distance;
  
  struct my3DVertexStruct temp = array[pivot_idx];
  array[pivot_idx] = array[right];
  array[right] = temp;
  
  int store_idx = left;
  for (int i = left; i < right; i++) {
    if (array[i].distance < pivot) {
      temp = array[i];
      array[i] = array[store_idx];
      array[store_idx] = temp;
      store_idx++;
    }
  }
  
  temp = array[store_idx];
  array[store_idx] = array[right];
  array[right] = temp;
  
  int pivot_final = store_idx;
  
#ifdef _OPENMP
  #pragma omp task shared(array) if(depth < 3)
#endif
  parallel_qsort(array, left, pivot_final - 1, depth + 1);
  
#ifdef _OPENMP
  #pragma omp task shared(array) if(depth < 3)
#endif
  parallel_qsort(array, pivot_final + 1, right, depth + 1);
  
#ifdef _OPENMP
  #pragma omp taskwait
#endif
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
      int x_sq = x * x;
      int y_sq = y * y;
      int z_sq = z * z;
      array[count].distance = sqrt((double)(x_sq + y_sq + z_sq));
      count++;
    }
    fclose(fp);
    fp = NULL;
  }
  
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);
#ifdef _OPENMP
  #pragma omp parallel
  {
    #pragma omp single
    {
      parallel_qsort(array, 0, count - 1, 0);
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
