/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* symm.c: this file is part of PolyBench/C */
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
#include "symm.h"


/* Array initialization. */
static
void init_array(int m, int n,
		DATA_TYPE *alpha,
		DATA_TYPE *beta,
		DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j;

  *alpha = 1.5;
  *beta = 1.2;
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
      C[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
      B[i][j] = (DATA_TYPE) ((n+i-j) % 100) / m;
    }
  for (i = 0; i < m; i++) {
    for (j = 0; j <=i; j++)
      A[i][j] = (DATA_TYPE) ((i+j) % 100) / m;
    for (j = i+1; j < m; j++)
      A[i][j] = -999; //regions of arrays that should not be used
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++) {
	if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_symm(int m, int n,
		 DATA_TYPE alpha,
		 DATA_TYPE beta,
		 DATA_TYPE POLYBENCH_2D(C,M,N,m,n),
		 DATA_TYPE POLYBENCH_2D(A,M,M,m,m),
		 DATA_TYPE POLYBENCH_2D(B,M,N,m,n))
{
  int i, j, k;
  DATA_TYPE temp2;
#pragma scop
  /* Parallelize outer i-loop. Each iteration updates a disjoint subset of rows
     in C: row i and rows 0..i-1. The latter are private per-i because the
     update is C[k][j] += alpha*B[i][j]*A[i][k], which is unique to (i,k,j). */
#pragma omp parallel for private(j,k,temp2) schedule(static)
  for (i = 0; i < _PB_M; i++)
  {
    const DATA_TYPE alpha_local = alpha;
    const DATA_TYPE beta_local  = beta;

    for (j = 0; j < _PB_N; j++ )
    {
      DATA_TYPE temp2_local = 0;

      /* Manually unroll the inner k-loop by factor 4 to reduce loop overhead
         and expose more instruction-level parallelism. */
      int k_lim = i & ~3;
      for (k = 0; k < k_lim; k += 4)
      {
        DATA_TYPE Bij = B[i][j];
        DATA_TYPE Aik0 = A[i][k];
        DATA_TYPE Aik1 = A[i][k+1];
        DATA_TYPE Aik2 = A[i][k+2];
        DATA_TYPE Aik3 = A[i][k+3];

        C[k][j]     += alpha_local * Bij * Aik0;
        C[k+1][j]   += alpha_local * Bij * Aik1;
        C[k+2][j]   += alpha_local * Bij * Aik2;
        C[k+3][j]   += alpha_local * Bij * Aik3;

        temp2_local += B[k][j]   * Aik0;
        temp2_local += B[k+1][j] * Aik1;
        temp2_local += B[k+2][j] * Aik2;
        temp2_local += B[k+3][j] * Aik3;
      }

      for (; k < i; k++)
      {
        C[k][j]     += alpha_local * B[i][j] * A[i][k];
        temp2_local += B[k][j] * A[i][k];
      }

      C[i][j] = beta_local * C[i][j]
              + alpha_local * B[i][j] * A[i][i]
              + alpha_local * temp2_local;
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
  int m = M;
  int n = N;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  DATA_TYPE beta;
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,M,N,m,n);
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,M,M,m,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,N,m,n);

  /* Initialize array(s). */
  init_array (m, n, &alpha, &beta,
	      POLYBENCH_ARRAY(C),
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B));

  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Run kernel. */
  kernel_symm (m, n,
	       alpha, beta,
	       POLYBENCH_ARRAY(C),
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(B));

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(C);
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