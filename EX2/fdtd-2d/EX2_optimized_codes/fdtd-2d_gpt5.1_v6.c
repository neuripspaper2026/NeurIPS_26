/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* fdtd-2d.c: this file is part of PolyBench/C */
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
#include "fdtd-2d.h"


/* Array initialization. */
static
void init_array (int tmax,
		 int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int i, j;

  for (i = 0; i < tmax; i++)
    _fict_[i] = (DATA_TYPE) i;
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      {
	ex[i][j] = ((DATA_TYPE) i*(j+1)) / nx;
	ey[i][j] = ((DATA_TYPE) i*(j+2)) / ny;
	hz[i][j] = ((DATA_TYPE) i*(j+3)) / nx;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("ex");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ex[i][j]);
    }
  POLYBENCH_DUMP_END("ex");
  POLYBENCH_DUMP_FINISH;

  POLYBENCH_DUMP_BEGIN("ey");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, ey[i][j]);
    }
  POLYBENCH_DUMP_END("ey");

  POLYBENCH_DUMP_BEGIN("hz");
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
      if ((i * nx + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, hz[i][j]);
    }
  POLYBENCH_DUMP_END("hz");
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_fdtd_2d(int tmax,
                    int nx,
                    int ny,
                    DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny),
                    DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny),
                    DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny),
                    DATA_TYPE POLYBENCH_1D(_fict_,TMAX,tmax))
{
  int t, i, j;
  const DATA_TYPE c0 = SCALAR_VAL(0.5);
  const DATA_TYPE c1 = SCALAR_VAL(0.7);

#pragma scop

  for (t = 0; t < _PB_TMAX; t++)
    {
      /* ey[0][j] depends only on t */
      DATA_TYPE fict_t = _fict_[t];

      /* Update ey row 0: independent over j */
      #pragma omp parallel for if(_PB_NY > 63) default(none) private(j) shared(ey,fict_t) schedule(static)
      for (j = 0; j < _PB_NY; j++)
        ey[0][j] = fict_t;

      /* Update ey for i >= 1: independent over (i,j) */
      #pragma omp parallel for if((_PB_NX * _PB_NY) > 4096) default(none) private(i,j) shared(ey,hz,c0) schedule(static)
      for (i = 1; i < _PB_NX; i++)
        {
          DATA_TYPE * __restrict ey_i = ey[i];
          DATA_TYPE * __restrict hz_i = hz[i];
          DATA_TYPE * __restrict hz_im1 = hz[i-1];
          for (j = 0; j < _PB_NY; j++)
            ey_i[j] = ey_i[j] - c0 * (hz_i[j] - hz_im1[j]);
        }

      /* Update ex for j >= 1: independent over (i,j) */
      #pragma omp parallel for if((_PB_NX * _PB_NY) > 4096) default(none) private(i,j) shared(ex,hz,c0) schedule(static)
      for (i = 0; i < _PB_NX; i++)
        {
          DATA_TYPE * __restrict ex_i = ex[i];
          DATA_TYPE * __restrict hz_i = hz[i];
          for (j = 1; j < _PB_NY; j++)
            ex_i[j] = ex_i[j] - c0 * (hz_i[j] - hz_i[j-1]);
        }

      /* Update hz: independent over (i,j), uses updated ex, ey of same t step */
      #pragma omp parallel for if((_PB_NX * _PB_NY) > 4096) default(none) private(i,j) shared(hz,ex,ey,c1) schedule(static)
      for (i = 0; i < _PB_NX - 1; i++)
        {
          DATA_TYPE * __restrict hz_i = hz[i];
          DATA_TYPE * __restrict ex_i = ex[i];
          DATA_TYPE * __restrict ex_ip1 = ex[i];   /* ex[i][j+1] uses same i; keep for clarity */
          DATA_TYPE * __restrict ey_i = ey[i];
          DATA_TYPE * __restrict ey_ip1 = ey[i+1];
          for (j = 0; j < _PB_NY - 1; j++)
            hz_i[j] = hz_i[j] - c1 * ((ex_ip1[j+1] - ex_i[j]) +
                                      (ey_ip1[j]   - ey_i[j]));
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
  int tmax = TMAX;
  int nx = NX;
  int ny = NY;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(ey,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_2D_ARRAY_DECL(hz,DATA_TYPE,NX,NY,nx,ny);
  POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,tmax);

  /* Initialize array(s). */
  init_array (tmax, nx, ny,
	      POLYBENCH_ARRAY(ex),
	      POLYBENCH_ARRAY(ey),
	      POLYBENCH_ARRAY(hz),
	      POLYBENCH_ARRAY(_fict_));

  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Run kernel. */
  kernel_fdtd_2d (tmax, nx, ny,
		  POLYBENCH_ARRAY(ex),
		  POLYBENCH_ARRAY(ey),
		  POLYBENCH_ARRAY(hz),
		  POLYBENCH_ARRAY(_fict_));


  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(ex),
				    POLYBENCH_ARRAY(ey),
				    POLYBENCH_ARRAY(hz)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(ex);
  POLYBENCH_FREE_ARRAY(ey);
  POLYBENCH_FREE_ARRAY(hz);
  POLYBENCH_FREE_ARRAY(_fict_);

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