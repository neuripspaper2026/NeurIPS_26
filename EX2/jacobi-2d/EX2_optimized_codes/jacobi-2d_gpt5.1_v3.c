/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-2d.c: this file is part of PolyBench/C */
#define POLYBENCH_DUMP_ARRAYS
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <omp.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-2d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
	B[i][j] = ((DATA_TYPE) i*(j+3) + 3) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_jacobi_2d(int tsteps,
                      int n,
                      DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
                      DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int t, i, j;

#pragma scop
  for (t = 0; t < _PB_TSTEPS; t++)
  {
    const int i_start = 1;
    const int i_end   = _PB_N - 1;
    const int j_start = 1;
    const int j_end   = _PB_N - 1;

    /* First sweep: compute B from A */
    for (i = i_start; i < i_end; i++)
    {
      DATA_TYPE *restrict Bi = B[i];
      DATA_TYPE *restrict Ai    = A[i];
      DATA_TYPE *restrict Ai_m1 = A[i-1];
      DATA_TYPE *restrict Ai_p1 = A[i+1];

      DATA_TYPE left   = Ai[j_start-1]; /* A[i][0] */
      DATA_TYPE center = Ai[j_start];   /* A[i][1] */
      DATA_TYPE right  = Ai[j_start+1]; /* A[i][2] */

      for (j = j_start; j < j_end; j++)
      {
        const DATA_TYPE new_center = SCALAR_VAL(0.2) *
          (center +
           left +
           right +
           Ai_m1[j] +
           Ai_p1[j]);

        Bi[j] = new_center;

        left   = center;
        center = right;
        if (j + 2 <= j_end)
          right = Ai[j+2];
      }
    }

    /* Second sweep: compute A from B */
    for (i = i_start; i < i_end; i++)
    {
      DATA_TYPE *restrict Ai = A[i];
      DATA_TYPE *restrict Bi    = B[i];
      DATA_TYPE *restrict Bi_m1 = B[i-1];
      DATA_TYPE *restrict Bi_p1 = B[i+1];

      DATA_TYPE left   = Bi[j_start-1]; /* B[i][0] */
      DATA_TYPE center = Bi[j_start];   /* B[i][1] */
      DATA_TYPE right  = Bi[j_start+1]; /* B[i][2] */

      for (j = j_start; j < j_end; j++)
      {
        const DATA_TYPE new_center = SCALAR_VAL(0.2) *
          (center +
           left +
           right +
           Bi_m1[j] +
           Bi_p1[j]);

        Ai[j] = new_center;

        left   = center;
        center = right;
        if (j + 2 <= j_end)
          right = Bi[j+2];
      }
    }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Start timing for total execution */
  struct timespec main_start, main_end;
  clock_gettime(CLOCK_MONOTONIC, &main_start);

  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Run kernel. */
  kernel_jacobi_2d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  /* End timing for total execution */
  clock_gettime(CLOCK_MONOTONIC, &main_end);
  double main_time = (main_end.tv_sec - main_start.tv_sec) +
                     (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

  /* Determine timing output destination */
  FILE *timing_file = stderr;
  const char *timing_path = getenv("TIMING_LOG_FILE");
  if (timing_path && timing_path[0] != '\0') {
    FILE *tmp = fopen(timing_path, "w");
    if (tmp)
      timing_file = tmp;
  }

  /* Print timing results */
  fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
  fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

  if (timing_file != stderr)
    fclose(timing_file);

  return 0;
}