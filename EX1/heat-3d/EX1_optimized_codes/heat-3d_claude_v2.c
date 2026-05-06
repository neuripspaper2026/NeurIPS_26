/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* heat-3d.c: this file is part of PolyBench/C */
#define POLYBENCH_DUMP_ARRAYS
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "heat-3d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		 DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int i, j, k;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++)
        A[i][j][k] = B[i][j][k] = (DATA_TYPE) (i + j + (n-k))* 10 / (n);
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n))

{
  int i, j, k;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++) {
         if ((i * n * n + j * n + k) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
         fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j][k]);
      }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_heat_3d(int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_3D(A,N,N,N,n,n,n),
		      DATA_TYPE POLYBENCH_3D(B,N,N,N,n,n,n))
{
  int t, i, j, k;
  const int n_minus_1 = n - 1;
  const DATA_TYPE c1 = SCALAR_VAL(0.125);
  const DATA_TYPE c2 = SCALAR_VAL(-0.25);

#pragma scop
    for (t = 1; t <= TSTEPS; t++) {
        for (i = 1; i < n_minus_1; i++) {
            DATA_TYPE (*restrict Ai)[N] = A[i];
            DATA_TYPE (*restrict Aim1)[N] = A[i-1];
            DATA_TYPE (*restrict Aip1)[N] = A[i+1];
            DATA_TYPE (*restrict Bi)[N] = B[i];
            
            for (j = 1; j < n_minus_1; j++) {
                DATA_TYPE *restrict Aij = Ai[j];
                DATA_TYPE *restrict Aijm1 = Ai[j-1];
                DATA_TYPE *restrict Aijp1 = Ai[j+1];
                DATA_TYPE *restrict Aim1j = Aim1[j];
                DATA_TYPE *restrict Aip1j = Aip1[j];
                DATA_TYPE *restrict Bij = Bi[j];
                
                for (k = 1; k < n_minus_1; k++) {
                    DATA_TYPE center = Aij[k];
                    Bij[k] = c1 * (Aip1j[k] + Aim1j[k] + Aijp1[k] + Aijm1[k] + Aij[k+1] + Aij[k-1]) + c2 * center + center;
                }
            }
        }
        
        for (i = 1; i < n_minus_1; i++) {
            DATA_TYPE (*restrict Ai)[N] = A[i];
            DATA_TYPE (*restrict Bi)[N] = B[i];
            DATA_TYPE (*restrict Bim1)[N] = B[i-1];
            DATA_TYPE (*restrict Bip1)[N] = B[i+1];
            
            for (j = 1; j < n_minus_1; j++) {
                DATA_TYPE *restrict Aij = Ai[j];
                DATA_TYPE *restrict Bij = Bi[j];
                DATA_TYPE *restrict Bijm1 = Bi[j-1];
                DATA_TYPE *restrict Bijp1 = Bi[j+1];
                DATA_TYPE *restrict Bim1j = Bim1[j];
                DATA_TYPE *restrict Bip1j = Bip1[j];
                
                for (k = 1; k < n_minus_1; k++) {
                    DATA_TYPE center = Bij[k];
                    Aij[k] = c1 * (Bip1j[k] + Bim1j[k] + Bijp1[k] + Bijm1[k] + Bij[k+1] + Bij[k-1]) + c2 * center + center;
                }
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
  POLYBENCH_3D_ARRAY_DECL(A, DATA_TYPE, N, N, N, n, n, n);
  POLYBENCH_3D_ARRAY_DECL(B, DATA_TYPE, N, N, N, n, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Run kernel. */
  kernel_heat_3d (tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);

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