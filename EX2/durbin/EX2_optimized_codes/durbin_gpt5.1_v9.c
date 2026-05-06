/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* durbin.c: this file is part of PolyBench/C */
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
#include "durbin.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      r[i] = (n+1-i);
    }
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
void kernel_durbin(int n,
		   DATA_TYPE POLYBENCH_1D(r,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n))
{
  DATA_TYPE z[N];
  DATA_TYPE alpha;
  DATA_TYPE beta;
  DATA_TYPE sum;

  int i, k;

#pragma scop
  y[0] = -r[0];
  beta = SCALAR_VAL(1.0);
  alpha = -r[0];

  for (k = 1; k < _PB_N; k++) {
    beta = (SCALAR_VAL(1.0) - alpha * alpha) * beta;
    sum = SCALAR_VAL(0.0);

    /* Parallel reduction on inner dot-product */
#ifdef _OPENMP
#pragma omp parallel for default(none) private(i) shared(r,y,k) reduction(+:sum)
#endif
    for (i = 0; i < k; i++) {
      sum += r[k - i - 1] * y[i];
    }

    alpha = - (r[k] + sum) / beta;

    /* Fuse z and y updates: write directly to y to avoid extra copy.
       This preserves the original recurrence because only y[0..k-1]
       are read, and they are all read before being overwritten. */
#ifdef _OPENMP
#pragma omp parallel for default(none) private(i) shared(y,k,alpha)
#endif
    for (i = 0; i < k; i++) {
      y[i] = y[i] + alpha * y[k - i - 1];
    }

    y[k] = alpha;
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

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(r));

  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Run kernel. */
  kernel_durbin (n,
		 POLYBENCH_ARRAY(r),
		 POLYBENCH_ARRAY(y));

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(y)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(r);
  POLYBENCH_FREE_ARRAY(y);

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