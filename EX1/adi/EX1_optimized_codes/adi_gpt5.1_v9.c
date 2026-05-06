/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* adi.c: this file is part of PolyBench/C */
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
#include "adi.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	u[i][j] =  (DATA_TYPE)(i + n-j) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(u,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("u");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, u[i][j]);
    }
  POLYBENCH_DUMP_END("u");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel Computers"
 * by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
static
void kernel_adi(int tsteps, int n,
		DATA_TYPE POLYBENCH_2D(u,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(v,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(p,N,N,n,n),
		DATA_TYPE POLYBENCH_2D(q,N,N,n,n))
{
  int t, i, j;
  DATA_TYPE DX, DY, DT;
  DATA_TYPE B1, B2;
  DATA_TYPE mul1, mul2;
  DATA_TYPE a, b, c, d, e, f;

#pragma scop

  DX = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DY = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  DT = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_TSTEPS;
  B1 = SCALAR_VAL(2.0);
  B2 = SCALAR_VAL(1.0);
  mul1 = B1 * DT / (DX * DX);
  mul2 = B2 * DT / (DY * DY);

  a = -mul1 /  SCALAR_VAL(2.0);
  b = SCALAR_VAL(1.0)+mul1;
  c = a;
  d = -mul2 / SCALAR_VAL(2.0);
  e = SCALAR_VAL(1.0)+mul2;
  f = d;

  {
    const DATA_TYPE one  = SCALAR_VAL(1.0);
    const DATA_TYPE two  = SCALAR_VAL(2.0);
    const DATA_TYPE zero = SCALAR_VAL(0.0);

    for (t = 1; t <= _PB_TSTEPS; t++) {
      /* Column Sweep */
      for (i = 1; i < _PB_N - 1; i++) {
        DATA_TYPE *restrict p_i = p[i];
        DATA_TYPE *restrict q_i = q[i];
        const DATA_TYPE *restrict u_colm1 = &u[0][i-1];
        const DATA_TYPE *restrict u_col   = &u[0][i];
        const DATA_TYPE *restrict u_colp1 = &u[0][i+1];
        DATA_TYPE *restrict v_col = &v[0][i];

        v_col[0] = one;
        p_i[0]   = zero;
        q_i[0]   = v_col[0];

        for (j = 1; j < _PB_N - 1; j++) {
          const DATA_TYPE denom = a * p_i[j-1] + b;
          const DATA_TYPE u_j_im1 = u_colm1[j];
          const DATA_TYPE u_j_i   = u_col[j];
          const DATA_TYPE u_j_ip1 = u_colp1[j];
          const DATA_TYPE q_prev  = q_i[j-1];

          p_i[j] = -c / denom;
          q_i[j] = (-d * u_j_im1 + (one + two * d) * u_j_i - f * u_j_ip1 - a * q_prev) / denom;
        }

        v_col[_PB_N - 1] = one;
        for (j = _PB_N - 2; j >= 1; j--) {
          v_col[j] = p_i[j] * v_col[j+1] + q_i[j];
        }
      }

      /* Row Sweep */
      for (i = 1; i < _PB_N - 1; i++) {
        DATA_TYPE *restrict u_i = u[i];
        DATA_TYPE *restrict p_i = p[i];
        DATA_TYPE *restrict q_i = q[i];
        const DATA_TYPE *restrict v_im1 = v[i-1];
        const DATA_TYPE *restrict v_i   = v[i];
        const DATA_TYPE *restrict v_ip1 = v[i+1];

        u_i[0] = one;
        p_i[0] = zero;
        q_i[0] = u_i[0];

        for (j = 1; j < _PB_N - 1; j++) {
          const DATA_TYPE denom = d * p_i[j-1] + e;
          const DATA_TYPE v_im1_j = v_im1[j];
          const DATA_TYPE v_i_j   = v_i[j];
          const DATA_TYPE v_ip1_j = v_ip1[j];
          const DATA_TYPE q_prev  = q_i[j-1];

          p_i[j] = -f / denom;
          q_i[j] = (-a * v_im1_j + (one + two * a) * v_i_j - c * v_ip1_j - d * q_prev) / denom;
        }

        u_i[_PB_N - 1] = one;
        for (j = _PB_N - 2; j >= 1; j--) {
          u_i[j] = p_i[j] * u_i[j+1] + q_i[j];
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
  POLYBENCH_2D_ARRAY_DECL(u, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(v, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(p, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(q, DATA_TYPE, N, N, n, n);


  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(u));

  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Run kernel. */
  kernel_adi (tsteps, n, POLYBENCH_ARRAY(u), POLYBENCH_ARRAY(v), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(u)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(u);
  POLYBENCH_FREE_ARRAY(v);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(q);

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