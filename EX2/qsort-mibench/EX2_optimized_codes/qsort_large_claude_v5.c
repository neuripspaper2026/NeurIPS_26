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

static void parallel_qsort(struct my3DVertexStruct *array, int count) {
#ifdef _OPENMP
  if (count < 10000) {
    qsort(array, count, sizeof(struct my3DVertexStruct), compare);
    return;
  }
  
  int pivot_idx = count / 2;
  double pivot = array[pivot_idx].distance;
  
  int left_count = 0;
  int right_count = 0;
  
  #pragma omp parallel
  {
    int local_left = 0;
    int local_right = 0;
    
    #pragma omp for nowait
    for (int i = 0; i < count; i++) {
      if (array[i].distance < pivot) local_left++;
      else if (array[i].distance > pivot) local_right++;
    }
    
    #pragma omp atomic
    left_count += local_left;
    
    #pragma omp atomic
    right_count += local_right;
  }
  
  struct my3DVertexStruct *temp = (struct my3DVertexStruct *)malloc(count * sizeof(struct my3DVertexStruct));
  if (!temp) {
    qsort(array, count, sizeof(struct my3DVertexStruct), compare);
    return;
  }
  
  int *left_indices = (int *)malloc(count * sizeof(int));
  int *right_indices = (int *)malloc(count * sizeof(int));
  int *equal_indices = (int *)malloc(count * sizeof(int));
  
  if (!left_indices || !right_indices || !equal_indices) {
    free(temp);
    free(left_indices);
    free(right_indices);
    free(equal_indices);
    qsort(array, count, sizeof(struct my3DVertexStruct), compare);
    return;
  }
  
  #pragma omp parallel
  {
    int local_left_idx = 0;
    int local_right_idx = 0;
    int local_equal_idx = 0;
    
    #pragma omp for nowait
    for (int i = 0; i < count; i++) {
      if (array[i].distance < pivot) {
        left_indices[local_left_idx++] = i;
      } else if (array[i].distance > pivot) {
        right_indices[local_right_idx++] = i;
      } else {
        equal_indices[local_equal_idx++] = i;
      }
    }
    
    #pragma omp single
    {
      int pos = 0;
      for (int i = 0; i < local_left_idx; i++) {
        temp[pos++] = array[left_indices[i]];
      }
      for (int i = 0; i < local_equal_idx; i++) {
        temp[pos++] = array[equal_indices[i]];
      }
      for (int i = 0; i < local_right_idx; i++) {
        temp[pos++] = array[right_indices[i]];
      }
    }
  }
  
  for (int i = 0; i < count; i++) {
    array[i] = temp[i];
  }
  
  free(temp);
  free(left_indices);
  free(right_indices);
  free(equal_indices);
  
  int equal_count = count - left_count - right_count;
  
  #pragma omp task shared(array) if(left_count > 10000)
  {
    if (left_count > 1) {
      parallel_qsort(array, left_count);
    }
  }
  
  #pragma omp task shared(array) if(right_count > 10000)
  {
    if (right_count > 1) {
      parallel_qsort(array + left_count + equal_count, right_count);
    }
  }
  
  #pragma omp taskwait
#else
  qsort(array, count, sizeof(struct my3DVertexStruct), compare);
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
      double dx = (double)x;
      double dy = (double)y;
      double dz = (double)z;
      array[count].distance = sqrt(dx*dx + dy*dy + dz*dz);
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
      parallel_qsort(array, count);
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
