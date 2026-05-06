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

#pragma scop
  {
    const DATA_TYPE c0 = SCALAR_VAL(0.125);
    const DATA_TYPE c1 = SCALAR_VAL(2.0);
    const int n_end = _PB_N - 1;
    const int t_end = TSTEPS;

    for (t = 1; t <= t_end; t++) {
      for (i = 1; i < n_end; i++) {
        DATA_TYPE (*Ai)[N][N] = A[i];
        DATA_TYPE (*Aip)[N][N] = A[i+1];
        DATA_TYPE (*Aim)[N][N] = A[i-1];
        DATA_TYPE (*Bi)[N][N] = B[i];

        for (j = 1; j < n_end; j++) {
          DATA_TYPE *restrict Ai_j   = Ai[j];
          DATA_TYPE *restrict Ai_jp  = Ai[j+1];
          DATA_TYPE *restrict Ai_jm  = Ai[j-1];
          DATA_TYPE *restrict Bi_j   = Bi[j];

          for (k = 1; k < n_end; k++) {
            const DATA_TYPE center = Ai_j[k];
            const DATA_TYPE x_diff = Aip[j][k] - c1 * center + Aim[j][k];
            const DATA_TYPE y_diff = Ai_jp[k]  - c1 * center + Ai_jm[k];
            const DATA_TYPE z_diff = Ai_j[k+1] - c1 * center + Ai_j[k-1];

            Bi_j[k] = center + c0 * (x_diff + y_diff + z_diff);
          }
        }
      }

      for (i = 1; i < n_end; i++) {
        DATA_TYPE (*Bi)[N][N]  = B[i];
        DATA_TYPE (*Bip)[N][N] = B[i+1];
        DATA_TYPE (*Bim)[N][N] = B[i-1];
        DATA_TYPE (*Ai)[N][N]  = A[i];

        for (j = 1; j < n_end; j++) {
          DATA_TYPE *restrict Bi_j   = Bi[j];
          DATA_TYPE *restrict Bi_jp  = Bi[j+1];
          DATA_TYPE *restrict Bi_jm  = Bi[j-1];
          DATA_TYPE *restrict Ai_j   = Ai[j];

          for (k = 1; k < n_end; k++) {
            const DATA_TYPE center = Bi_j[k];
            const DATA_TYPE x_diff = Bip[j][k] - c1 * center + Bim[j][k];
            const DATA_TYPE y_diff = Bi_jp[k]  - c1 * center + Bi_jm[k];
            const DATA_TYPE z_diff = Bi_j[k+1] - c1 * center + Bi_j[k-1];

            Ai_j[k] = center + c0 * (x_diff + y_diff + z_diff);
          }
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