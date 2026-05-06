/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* bicg.c: this file is part of PolyBench/C */
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
#include "bicg.h"


/* Array initialization. */
static
void init_array (int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m))
{
  int i, j;

  for (i = 0; i < m; i++)
    p[i] = (DATA_TYPE)(i % m) / m;
  for (i = 0; i < n; i++) {
    r[i] = (DATA_TYPE)(i % n) / n;
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) (i*(j+1) % n)/n;
  }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int m, int n,
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n))

{
  int i;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("s");
  for (i = 0; i < m; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, s[i]);
  }
  POLYBENCH_DUMP_END("s");
  POLYBENCH_DUMP_BEGIN("q");
  for (i = 0; i < n; i++) {
    if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
    fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, q[i]);
  }
  POLYBENCH_DUMP_END("q");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_bicg(int m, int n,
		 DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		 DATA_TYPE POLYBENCH_1D(s,M,m),
		 DATA_TYPE POLYBENCH_1D(q,N,n),
		 DATA_TYPE POLYBENCH_1D(p,M,m),
		 DATA_TYPE POLYBENCH_1D(r,N,n))
{
  int i, j;

#pragma scop
  /* Initialize s to zero (parallelizable, no dependencies). */
#pragma omp parallel for if (_PB_M > 256)
  for (i = 0; i < _PB_M; i++)
    s[i] = SCALAR_VAL(0.0);

  /* Main computation: q[i] and s[j].  We introduce a temporary
     private buffer for s to avoid write conflicts and then
     accumulate into the global s in a second step. */

  /* Thread-private partial sums of s. */
  DATA_TYPE (*s_priv)[M + POLYBENCH_PADDING_FACTOR] = NULL;

#ifdef _OPENMP
  {
    int num_threads = 1;
#pragma omp parallel
    {
      int tid_local, nt_local, i_local, j_local;

#pragma omp single
      {
        num_threads = omp_get_num_threads();
        s_priv = (DATA_TYPE (*)[M + POLYBENCH_PADDING_FACTOR])
                 calloc((size_t)num_threads * (size_t)M, sizeof(DATA_TYPE));
      }

#pragma omp barrier

      tid_local = omp_get_thread_num();
      nt_local  = num_threads;

#pragma omp for schedule(static)
      for (i_local = 0; i_local < _PB_N; i_local++)
      {
        DATA_TYPE qi = SCALAR_VAL(0.0);
        DATA_TYPE ri = r[i_local];
        DATA_TYPE *s_loc = &s_priv[tid_local][0];

        for (j_local = 0; j_local < _PB_M; j_local++)
        {
          DATA_TYPE Aij = A[i_local][j_local];
          s_loc[j_local] += ri * Aij;
          qi += Aij * p[j_local];
        }
        q[i_local] = qi;
      }

#pragma omp for schedule(static)
      for (j_local = 0; j_local < _PB_M; j_local++)
      {
        DATA_TYPE sum = SCALAR_VAL(0.0);
        for (i_local = 0; i_local < nt_local; i_local++)
          sum += s_priv[i_local][j_local];
        s[j_local] += sum;
      }

#pragma omp barrier
#pragma omp single
      {
        free(s_priv);
        s_priv = NULL;
      }
    }
  }
#else
  /* Serial fallback (no OpenMP). */
  for (i = 0; i < _PB_N; i++)
    {
      DATA_TYPE qi = SCALAR_VAL(0.0);
      DATA_TYPE ri = r[i];
      for (j = 0; j < _PB_M; j++)
        {
          DATA_TYPE Aij = A[i][j];
          s[j] += ri * Aij;
          qi   += Aij * p[j];
        }
      q[i] = qi;
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
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, M, n, m);
  POLYBENCH_1D_ARRAY_DECL(s, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(q, DATA_TYPE, N, n);
  POLYBENCH_1D_ARRAY_DECL(p, DATA_TYPE, M, m);
  POLYBENCH_1D_ARRAY_DECL(r, DATA_TYPE, N, n);

  /* Initialize array(s). */
  init_array (m, n,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(r),
	      POLYBENCH_ARRAY(p));

  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Run kernel. */
  kernel_bicg (m, n,
	       POLYBENCH_ARRAY(A),
	       POLYBENCH_ARRAY(s),
	       POLYBENCH_ARRAY(q),
	       POLYBENCH_ARRAY(p),
	       POLYBENCH_ARRAY(r));

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(m, n, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(q)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(s);
  POLYBENCH_FREE_ARRAY(q);
  POLYBENCH_FREE_ARRAY(p);
  POLYBENCH_FREE_ARRAY(r);

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