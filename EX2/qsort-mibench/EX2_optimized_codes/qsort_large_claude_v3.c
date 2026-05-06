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
  
  int len = right - left + 1;
  if (len < 10000 || depth > 3) {
    qsort(array + left, len, sizeof(struct my3DVertexStruct), compare);
    return;
  }
  
  int pivot_idx = left + len / 2;
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
  
  #pragma omp task shared(array) if(depth < 3)
  parallel_qsort(array, left, store_idx - 1, depth + 1);
  
  #pragma omp task shared(array) if(depth < 3)
  parallel_qsort(array, store_idx + 1, right, depth + 1);
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
    
    int capacity = MAXARRAY;
    int buffer_size = 65536;
    int *x_buf = (int *)malloc(buffer_size * sizeof(int));
    int *y_buf = (int *)malloc(buffer_size * sizeof(int));
    int *z_buf = (int *)malloc(buffer_size * sizeof(int));
    int buf_count = 0;
    
    if (x_buf && y_buf && z_buf) {
      while((fscanf(fp, "%d", &x) == 1) && (fscanf(fp, "%d", &y) == 1) && (fscanf(fp, "%d", &z) == 1) && (count < capacity)) {
        x_buf[buf_count] = x;
        y_buf[buf_count] = y;
        z_buf[buf_count] = z;
        buf_count++;
        
        if (buf_count == buffer_size) {
          #pragma omp parallel for schedule(static)
          for (int j = 0; j < buf_count; j++) {
            int xi = x_buf[j];
            int yi = y_buf[j];
            int zi = z_buf[j];
            double xi_d = (double)xi;
            double yi_d = (double)yi;
            double zi_d = (double)zi;
            array[count - buf_count + j].x = xi;
            array[count - buf_count + j].y = yi;
            array[count - buf_count + j].z = zi;
            array[count - buf_count + j].distance = sqrt(xi_d * xi_d + yi_d * yi_d + zi_d * zi_d);
          }
          count += buf_count;
          buf_count = 0;
        }
      }
      
      if (buf_count > 0) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < buf_count; j++) {
          int xi = x_buf[j];
          int yi = y_buf[j];
          int zi = z_buf[j];
          double xi_d = (double)xi;
          double yi_d = (double)yi;
          double zi_d = (double)zi;
          array[count + j].x = xi;
          array[count + j].y = yi;
          array[count + j].z = zi;
          array[count + j].distance = sqrt(xi_d * xi_d + yi_d * yi_d + zi_d * zi_d);
        }
        count += buf_count;
      }
      
      free(x_buf);
      free(y_buf);
      free(z_buf);
    } else {
      if (x_buf) free(x_buf);
      if (y_buf) free(y_buf);
      if (z_buf) free(z_buf);
      
      while((fscanf(fp, "%d", &x) == 1) && (fscanf(fp, "%d", &y) == 1) && (fscanf(fp, "%d", &z) == 1) && (count < capacity)) {
        array[count].x = x;
        array[count].y = y;
        array[count].z = z;
        double xd = (double)x;
        double yd = (double)y;
        double zd = (double)z;
        array[count].distance = sqrt(xd * xd + yd * yd + zd * zd);
        count++;
      }
    }
    
    fclose(fp);
    fp = NULL;
  }
  
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);
  #pragma omp parallel
  {
    #pragma omp single
    parallel_qsort(array, 0, count - 1, 0);
  }
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
