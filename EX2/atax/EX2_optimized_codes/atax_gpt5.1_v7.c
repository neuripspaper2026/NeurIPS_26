/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* atax.c: this file is part of PolyBench/C */
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
#include "atax.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
		 DATA_TYPE POLYBENCH_1D(x,N,n))
{
  int i, j;
  DATA_TYPE fn;
  fn = (DATA_TYPE)n;

  for (i = 0; i < n; i++)
      x[i] = 1 + (i / fn);
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % n) / (5*m);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_1D(y,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("y");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, y[i]);
  }
  POLYBENCH_DUMP_END("y");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_atax(int m, int n,
                 DATA_TYPE POLYBENCH_2D(A,M,N,m,n),
                 DATA_TYPE POLYBENCH_1D(x,N,n),
                 DATA_TYPE POLYBENCH_1D(y,N,n),
                 DATA_TYPE POLYBENCH_1D(tmp,M,m))
{
  int i, j;

#pragma scop
  /* Initialize y */
  for (i = 0; i < _PB_N; i++)
    y[i] = SCALAR_VAL(0.0);

  /* First phase: tmp[i] = A[i][*] * x[*] */
  for (i = 0; i < _PB_M; i++)
  {
    DATA_TYPE sum = SCALAR_VAL(0.0);
    DATA_TYPE * __restrict Ai = A[i];
    DATA_TYPE * __restrict xptr = x;
    for (j = 0; j < _PB_N; j++)
      sum += Ai[j] * xptr[j];
    tmp[i] = sum;
  }

  /* Second phase: y[j] += A[i][j] * tmp[i] */
  for (i = 0; i < _PB_M; i++)
  {
    DATA_TYPE ti = tmp[i];
    DATA_TYPE * __restrict Ai = A[i];
    for (j = 0; j < _PB_N; j++)
      y[j] += Ai[j] * ti;
  }

#ifdef _OPENMP
  /* Parallel optimized version */
  {
    int i2, j2;

    /* Re-initialize y in parallel */
#pragma omp parallel for schedule(static)
    for (j2 = 0; j2 < _PB_N; j2++)
      y[j2] = SCALAR_VAL(0.0);

    /* Parallel first phase: tmp[i] = A[i][*] * x[*] */
#pragma omp parallel for private(j2) schedule(static)
    for (i2 = 0; i2 < _PB_M; i2++)
    {
      DATA_TYPE sum2 = SCALAR_VAL(0.0);
      DATA_TYPE * __restrict Ai2 = A[i2];
      DATA_TYPE * __restrict xptr2 = x;
#pragma GCC ivdep
      for (j2 = 0; j2 < _PB_N; j2++)
        sum2 += Ai2[j2] * xptr2[j2];
      tmp[i2] = sum2;
    }

    /* Parallel second phase: reduction on y */
#pragma omp parallel
    {
      DATA_TYPE * __restrict local_y = (DATA_TYPE*)alloca(N * sizeof(DATA_TYPE));
      int jj;

      for (jj = 0; jj < _PB_N; jj++)
        local_y[jj] = SCALAR_VAL(0.0);

#pragma omp for schedule(static)
      for (i2 = 0; i2 < _PB_M; i2++)
      {
        DATA_TYPE ti2 = tmp[i2];
        DATA_TYPE * __restrict Ai2 = A[i2];
#pragma GCC ivdep
        for (j2 = 0; j2 < _PB_N; j2++)
          local_y[j2] += Ai2[j2] * ti2;
      }

#pragma omp critical
      {
        for (jj = 0; jj < _PB_N; jj++)
          y[jj] += local_y[jj];
      }
    }
  }
#endif

#pragma endscop

}


int main(int argc, char** argv)
{
  /* Start timing for total execution */
  struct timespec main_start, main_end;
  clock_gettime(CLOCK_MONOTONIC, &main_start);

  /* Retrieve problem size. */
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, M, N, m, n);
  POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(tmp, DATA_TYPE, M, m);

  /* Initialize array(s). */
  init_array (m, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(x));

  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Run kernel. */
  kernel_atax (m, n,
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(x),
	       POLYBENCH_ARRAY(y),
	       POLYBENCH_ARRAY(tmp));

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(x);
  POLYBENCH_FREE_ARRAY(y);
  POLYBENCH_FREE_ARRAY(tmp);

  /* End timing for total execution */
  clock_gettime(CLOCK_MONOTONIC, &main_end);
  double main_time = (main_end.tv_sec - main_start.tv_sec) +
                     (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

  /* Determine timing output destination */
  FILE *timing_file = stderr;
  const char *timing_path = getenv("TIMING_LOG_FILE");
  if (timing_path && timing_path[0] != '\0') {
    FILE *tmp_file = fopen(timing_path, "w");
    if (tmp_file)
      timing_file = tmp_file;
  }

  /* Print timing results */
  fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
  fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

  if (timing_file != stderr)
    fclose(timing_file);

  return 0;
}