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
#pragma omp parallel for private(i,j,k) schedule(dynamic,16) if(m>=64)
   for (i = 0; i < _PB_M; i++) {
      DATA_TYPE temp2;
      DATA_TYPE alpha_Bii = alpha * B[i][i];
      DATA_TYPE alpha_Aii = alpha * A[i][i];
      DATA_TYPE beta_local = beta;
      
      for (j = 0; j < _PB_N; j += 4) {
        DATA_TYPE temp2_0 = 0;
        DATA_TYPE temp2_1 = 0;
        DATA_TYPE temp2_2 = 0;
        DATA_TYPE temp2_3 = 0;
        
        DATA_TYPE alpha_Bij_0 = alpha * B[i][j];
        DATA_TYPE alpha_Bij_1 = (j+1 < _PB_N) ? alpha * B[i][j+1] : 0;
        DATA_TYPE alpha_Bij_2 = (j+2 < _PB_N) ? alpha * B[i][j+2] : 0;
        DATA_TYPE alpha_Bij_3 = (j+3 < _PB_N) ? alpha * B[i][j+3] : 0;
        
        for (k = 0; k < i; k++) {
           DATA_TYPE Aik = A[i][k];
           C[k][j] += alpha_Bij_0 * Aik;
           temp2_0 += B[k][j] * Aik;
           
           if (j+1 < _PB_N) {
             C[k][j+1] += alpha_Bij_1 * Aik;
             temp2_1 += B[k][j+1] * Aik;
           }
           if (j+2 < _PB_N) {
             C[k][j+2] += alpha_Bij_2 * Aik;
             temp2_2 += B[k][j+2] * Aik;
           }
           if (j+3 < _PB_N) {
             C[k][j+3] += alpha_Bij_3 * Aik;
             temp2_3 += B[k][j+3] * Aik;
           }
        }
        
        C[i][j] = beta_local * C[i][j] + alpha_Bij_0 * A[i][i] + alpha * temp2_0;
        if (j+1 < _PB_N)
          C[i][j+1] = beta_local * C[i][j+1] + alpha_Bij_1 * A[i][i] + alpha * temp2_1;
        if (j+2 < _PB_N)
          C[i][j+2] = beta_local * C[i][j+2] + alpha_Bij_2 * A[i][i] + alpha * temp2_2;
        if (j+3 < _PB_N)
          C[i][j+3] = beta_local * C[i][j+3] + alpha_Bij_3 * A[i][i] + alpha * temp2_3;
     }
     
     for (j = ((_PB_N/4)*4); j < _PB_N; j++) {
        temp2 = 0;
        DATA_TYPE alpha_Bij = alpha * B[i][j];
        for (k = 0; k < i; k++) {
           DATA_TYPE Aik = A[i][k];
           C[k][j] += alpha_Bij * Aik;
           temp2 += B[k][j] * Aik;
        }
        C[i][j] = beta * C[i][j] + alpha_Bij * A[i][i] + alpha * temp2;
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