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
  DATA_TYPE recip_DX, recip_DY, recip_TSTEPS;
  DATA_TYPE half_B1, half_B2;
  DATA_TYPE tmp1, tmp2, tmp3;
  DATA_TYPE p_ij_prev, q_ij_prev;
  DATA_TYPE p_denom_col, p_denom_row;
  DATA_TYPE u_j_i_minus_1, u_j_i, u_j_i_plus_1;
  DATA_TYPE v_i_minus_1_j, v_i_j, v_i_plus_1_j;

#pragma scop

  recip_DX = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  recip_DY = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_N;
  recip_TSTEPS = SCALAR_VAL(1.0)/(DATA_TYPE)_PB_TSTEPS;
  B1 = SCALAR_VAL(2.0);
  B2 = SCALAR_VAL(1.0);
  half_B1 = B1 * SCALAR_VAL(0.5);
  half_B2 = B2 * SCALAR_VAL(0.5);
  mul1 = half_B1 * recip_TSTEPS / (recip_DX * recip_DX);
  mul2 = half_B2 * recip_TSTEPS / (recip_DY * recip_DY);

  a = -mul1;
  b = SCALAR_VAL(1.0)+mul1+mul1;
  c = a;
  d = -mul2;
  e = SCALAR_VAL(1.0)+mul2+mul2;
  f = d;

  for (t=1; t<=_PB_TSTEPS; t++) {
    //Column Sweep
    for (i=1; i<_PB_N-1; i++) {
      v[0][i] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = v[0][i];
      p_ij_prev = p[i][0];
      q_ij_prev = q[i][0];
      p_denom_col = a * p_ij_prev + b;
      
      for (j=1; j<_PB_N-1; j++) {
        p[i][j] = -c / p_denom_col;
        u_j_i_minus_1 = u[j][i-1];
        u_j_i = u[j][i];
        u_j_i_plus_1 = u[j][i+1];
        tmp1 = -d * u_j_i_minus_1;
        tmp2 = (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*d) * u_j_i;
        tmp3 = -f * u_j_i_plus_1;
        q[i][j] = (tmp1 + tmp2 + tmp3 - a * q_ij_prev) / p_denom_col;
        
        p_ij_prev = p[i][j];
        q_ij_prev = q[i][j];
        p_denom_col = a * p_ij_prev + b;
      }

      v[_PB_N-1][i] = SCALAR_VAL(1.0);
      for (j=_PB_N-2; j>=1; j--) {
        v[j][i] = p[i][j] * v[j+1][i] + q[i][j];
      }
    }
    
    //Row Sweep
    for (i=1; i<_PB_N-1; i++) {
      u[i][0] = SCALAR_VAL(1.0);
      p[i][0] = SCALAR_VAL(0.0);
      q[i][0] = u[i][0];
      p_ij_prev = p[i][0];
      q_ij_prev = q[i][0];
      p_denom_row = d * p_ij_prev + e;
      
      for (j=1; j<_PB_N-1; j++) {
        p[i][j] = -f / p_denom_row;
        v_i_minus_1_j = v[i-1][j];
        v_i_j = v[i][j];
        v_i_plus_1_j = v[i+1][j];
        tmp1 = -a * v_i_minus_1_j;
        tmp2 = (SCALAR_VAL(1.0)+SCALAR_VAL(2.0)*a) * v_i_j;
        tmp3 = -c * v_i_plus_1_j;
        q[i][j] = (tmp1 + tmp2 + tmp3 - d * q_ij_prev) / p_denom_row;
        
        p_ij_prev = p[i][j];
        q_ij_prev = q[i][j];
        p_denom_row = d * p_ij_prev + e;
      }
      u[i][_PB_N-1] = SCALAR_VAL(1.0);
      for (j=_PB_N-2; j>=1; j--) {
        u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
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