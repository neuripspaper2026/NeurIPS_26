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

  {
    const DATA_TYPE inv_two = SCALAR_VAL(0.5);
    a = -mul1 * inv_two;
    b = SCALAR_VAL(1.0) + mul1;
    c = a;
    d = -mul2 * inv_two;
    e = SCALAR_VAL(1.0) + mul2;
    f = d;
  }

  for (t = 1; t <= _PB_TSTEPS; t++) {
    const DATA_TYPE one = SCALAR_VAL(1.0);
    const DATA_TYPE zero = SCALAR_VAL(0.0);
    const int n1 = _PB_N - 1;
    const int n2 = _PB_N - 2;

    /* Precompute constants used in inner loops */
    const DATA_TYPE two_d = SCALAR_VAL(2.0) * d;
    const DATA_TYPE one_plus_two_d = one + two_d;
    const DATA_TYPE two_a = SCALAR_VAL(2.0) * a;
    const DATA_TYPE one_plus_two_a = one + two_a;

    /* Column Sweep */
    for (i = 1; i < n1; i++) {
      DATA_TYPE *restrict v_col0 = &v[0][0];
      DATA_TYPE *restrict v_col_last = &v[n1][0];
      DATA_TYPE *restrict p_row = &p[i][0];
      DATA_TYPE *restrict q_row = &q[i][0];

      v_col0[i] = one;
      p_row[0] = zero;
      q_row[0] = v_col0[i];

      {
        DATA_TYPE prev_p = p_row[0];
        DATA_TYPE prev_q = q_row[0];

        for (j = 1; j < n1; j++) {
          const DATA_TYPE denom = a * prev_p + b;
          const DATA_TYPE inv_denom = SCALAR_VAL(1.0) / denom;

          DATA_TYPE *restrict u_col_im1 = &u[0][i - 1];
          DATA_TYPE *restrict u_col_i   = &u[0][i];
          DATA_TYPE *restrict u_col_ip1 = &u[0][i + 1];

          const DATA_TYPE uj_im1 = u_col_im1[j];
          const DATA_TYPE uj     = u_col_i[j];
          const DATA_TYPE uj_ip1 = u_col_ip1[j];

          const DATA_TYPE num = (-d * uj_im1 + one_plus_two_d * uj - f * uj_ip1 - a * prev_q);

          const DATA_TYPE pj = -c * inv_denom;
          const DATA_TYPE qj = num * inv_denom;

          p_row[j] = pj;
          q_row[j] = qj;

          prev_p = pj;
          prev_q = qj;
        }
      }

      v_col_last[i] = one;

      {
        DATA_TYPE *restrict v_col = &v[0][i];
        DATA_TYPE next_v = v_col[n1];

        for (j = n2; j >= 1; j--) {
          const DATA_TYPE pj = p_row[j];
          const DATA_TYPE qj = q_row[j];
          const DATA_TYPE vj = pj * next_v + qj;

          v_col[j] = vj;
          next_v = vj;
        }
      }
    }

    /* Row Sweep */
    for (i = 1; i < n1; i++) {
      DATA_TYPE *restrict u_row = &u[i][0];
      DATA_TYPE *restrict p_row = &p[i][0];
      DATA_TYPE *restrict q_row = &q[i][0];

      u_row[0] = one;
      p_row[0] = zero;
      q_row[0] = u_row[0];

      {
        DATA_TYPE prev_p = p_row[0];
        DATA_TYPE prev_q = q_row[0];

        for (j = 1; j < n1; j++) {
          const DATA_TYPE denom = d * prev_p + e;
          const DATA_TYPE inv_denom = SCALAR_VAL(1.0) / denom;

          DATA_TYPE *restrict v_row_im1 = &v[i - 1][0];
          DATA_TYPE *restrict v_row_i   = &v[i][0];
          DATA_TYPE *restrict v_row_ip1 = &v[i + 1][0];

          const DATA_TYPE v_im1j = v_row_im1[j];
          const DATA_TYPE v_ij   = v_row_i[j];
          const DATA_TYPE v_ip1j = v_row_ip1[j];

          const DATA_TYPE num = (-a * v_im1j + one_plus_two_a * v_ij - c * v_ip1j - d * prev_q);

          const DATA_TYPE pj = -f * inv_denom;
          const DATA_TYPE qj = num * inv_denom;

          p_row[j] = pj;
          q_row[j] = qj;

          prev_p = pj;
          prev_q = qj;
        }
      }

      u_row[n1] = one;

      {
        DATA_TYPE next_u = u_row[n1];

        for (j = n2; j >= 1; j--) {
          const DATA_TYPE pj = p_row[j];
          const DATA_TYPE qj = q_row[j];
          const DATA_TYPE uj = pj * next_u + qj;

          u_row[j] = uj;
          next_u = uj;
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