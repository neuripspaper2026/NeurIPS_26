#define POLYBENCH_DUMP_ARRAYS
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <stdlib.h>

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
  const int n_pb = _PB_N;
  DATA_TYPE *restrict path_base = &path[0][0];

#pragma scop
  for (k = 0; k < n_pb; k++)
  {
    const DATA_TYPE *restrict path_k = &path[k][0];

    for (i = 0; i < n_pb; i++)
    {
      DATA_TYPE *restrict path_ik = &path[i][0];
      const DATA_TYPE aik = path_ik[k];
      const DATA_TYPE *restrict path_i = &path[i][0];

      for (j = 0; j < n_pb; j++)
      {
        DATA_TYPE *restrict pij = path_base + (size_t)i * (size_t)n_pb + (size_t)j;
        const DATA_TYPE via_k = aik + path_k[j];
        const DATA_TYPE curr  = *pij;
        *pij = (curr < via_k) ? curr : via_k;
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