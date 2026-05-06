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
#pragma scop
#pragma omp parallel for private(i, j, k) schedule(dynamic, 16)
   for (i = 0; i < _PB_M; i++) {
      DATA_TYPE A_i_i = A[i][i];
      DATA_TYPE alpha_A_i_i = alpha * A_i_i;
      
      for (j = 0; j < _PB_N; j += 4) {
        DATA_TYPE temp2_0 = 0;
        DATA_TYPE temp2_1 = 0;
        DATA_TYPE temp2_2 = 0;
        DATA_TYPE temp2_3 = 0;
        
        DATA_TYPE B_i_j0 = B[i][j];
        DATA_TYPE B_i_j1 = (j+1 < _PB_N) ? B[i][j+1] : 0;
        DATA_TYPE B_i_j2 = (j+2 < _PB_N) ? B[i][j+2] : 0;
        DATA_TYPE B_i_j3 = (j+3 < _PB_N) ? B[i][j+3] : 0;
        
        DATA_TYPE alpha_B_i_j0 = alpha * B_i_j0;
        DATA_TYPE alpha_B_i_j1 = alpha * B_i_j1;
        DATA_TYPE alpha_B_i_j2 = alpha * B_i_j2;
        DATA_TYPE alpha_B_i_j3 = alpha * B_i_j3;
        
        for (k = 0; k < i; k++) {
           DATA_TYPE A_i_k = A[i][k];
           DATA_TYPE alpha_A_i_k = alpha * A_i_k;
           
           C[k][j] += alpha_B_i_j0 * A_i_k;
           temp2_0 += B[k][j] * A_i_k;
           
           if (j+1 < _PB_N) {
              C[k][j+1] += alpha_B_i_j1 * A_i_k;
              temp2_1 += B[k][j+1] * A_i_k;
           }
           
           if (j+2 < _PB_N) {
              C[k][j+2] += alpha_B_i_j2 * A_i_k;
              temp2_2 += B[k][j+2] * A_i_k;
           }
           
           if (j+3 < _PB_N) {
              C[k][j+3] += alpha_B_i_j3 * A_i_k;
              temp2_3 += B[k][j+3] * A_i_k;
           }
        }
        
        C[i][j] = beta * C[i][j] + alpha_B_i_j0 * A_i_i + alpha * temp2_0;
        
        if (j+1 < _PB_N)
           C[i][j+1] = beta * C[i][j+1] + alpha_B_i_j1 * A_i_i + alpha * temp2_1;
        
        if (j+2 < _PB_N)
           C[i][j+2] = beta * C[i][j+2] + alpha_B_i_j2 * A_i_i + alpha * temp2_2;
        
        if (j+3 < _PB_N)
           C[i][j+3] = beta * C[i][j+3] + alpha_B_i_j3 * A_i_i + alpha * temp2_3;
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