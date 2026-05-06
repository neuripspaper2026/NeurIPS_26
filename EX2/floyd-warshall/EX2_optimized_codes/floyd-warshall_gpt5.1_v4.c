#define POLYBENCH_DUMP_ARRAYS
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>

/* Include polybench common header. */
#include "../../../utilities/polybench.h"

/* Include benchmark-specific header. */
#include "../floyd-warshall.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      path[i][j] = i*j%7+1;
      if ((i+j)%13 == 0 || (i+j)%7==0 || (i+j)%11 == 0)
         path[i][j] = 999;
    }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(path,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("path");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, path[i][j]);
    }
  POLYBENCH_DUMP_END("path");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_floyd_warshall(int n,
			   DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j, k;

#pragma scop
  for (k = 0; k < _PB_N; k++)
    {
      DATA_TYPE *restrict path_k = path[k];

      /* Parallelize outer i-loop for each fixed k.
         Use static schedule for better load balance on regular loops. */
#ifdef _OPENMP
#pragma omp parallel for private(j) schedule(static)
#endif
      for (i = 0; i < _PB_N; i++)
        {
          DATA_TYPE *restrict path_i = path[i];
          const DATA_TYPE pik = path_i[k];

          /* Manually unroll j-loop to improve ILP and reduce branch cost. */
          int j_unrolled = _PB_N & ~3;
          for (j = 0; j < j_unrolled; j += 4)
            {
              DATA_TYPE v0 = path_i[j];
              DATA_TYPE v1 = path_i[j+1];
              DATA_TYPE v2 = path_i[j+2];
              DATA_TYPE v3 = path_i[j+3];

              DATA_TYPE alt0 = pik + path_k[j];
              DATA_TYPE alt1 = pik + path_k[j+1];
              DATA_TYPE alt2 = pik + path_k[j+2];
              DATA_TYPE alt3 = pik + path_k[j+3];

              path_i[j]   = (v0 < alt0) ? v0 : alt0;
              path_i[j+1] = (v1 < alt1) ? v1 : alt1;
              path_i[j+2] = (v2 < alt2) ? v2 : alt2;
              path_i[j+3] = (v3 < alt3) ? v3 : alt3;
            }

          /* Handle remaining iterations (if N not multiple of 4). */
          for (; j < _PB_N; j++)
            {
              DATA_TYPE v = path_i[j];
              DATA_TYPE alt = pik + path_k[j];
              path_i[j] = (v < alt) ? v : alt;
            }
        }
    }
#pragma endscop

}


int main(int argc, char** argv)
{
  struct timespec main_start, main_end;
  struct timespec kernel_start, kernel_end;
  double kernel_time, main_time;
  FILE *timing_file = stderr;
  const char *timing_path = getenv("TIMING_LOG_FILE");

  clock_gettime(CLOCK_MONOTONIC, &main_start);

  /* Retrieve problem size. */
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(path, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(path));

  /* Run kernel with timing. */
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);
  kernel_floyd_warshall (n, POLYBENCH_ARRAY(path));
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(path)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(path);

  clock_gettime(CLOCK_MONOTONIC, &main_end);
  main_time = (main_end.tv_sec - main_start.tv_sec) +
              (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

  if (timing_path && timing_path[0] != '\0') {
    FILE *tmp = fopen(timing_path, "w");
    if (tmp)
      timing_file = tmp;
  }

  fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
  fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

  if (timing_file != stderr)
    fclose(timing_file);

  return 0;
}