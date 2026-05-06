/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* deriche.c: this file is part of PolyBench/C */
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
#include "deriche.h"


/* Array initialization. */
static
void init_array (int w, int h, DATA_TYPE* alpha,
		 DATA_TYPE POLYBENCH_2D(imgIn,W,H,w,h),
		 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))
{
  int i, j;

  *alpha=0.25; //parameter of the filter

  //input should be between 0 and 1 (grayscale image pixel)
  for (i = 0; i < w; i++)
     for (j = 0; j < h; j++)
	imgIn[i][j] = (DATA_TYPE) ((313*i+991*j)%65536) / 65535.0f;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int w, int h,
		 DATA_TYPE POLYBENCH_2D(imgOut,W,H,w,h))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("imgOut");
  for (i = 0; i < w; i++)
    for (j = 0; j < h; j++) {
      if ((i * h + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, imgOut[i][j]);
    }
  POLYBENCH_DUMP_END("imgOut");
  POLYBENCH_DUMP_FINISH;
}



/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Original code provided by Gael Deest */
static
void kernel_deriche(int w, int h, DATA_TYPE alpha,
       DATA_TYPE POLYBENCH_2D(imgIn, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(imgOut, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y1, W, H, w, h),
       DATA_TYPE POLYBENCH_2D(y2, W, H, w, h)) {
    int i,j;
    DATA_TYPE xm1, tm1, ym1, ym2;
    DATA_TYPE xp1, xp2;
    DATA_TYPE tp1, tp2;
    DATA_TYPE yp1, yp2;

    const DATA_TYPE zero = SCALAR_VAL(0.0);
    const DATA_TYPE one  = SCALAR_VAL(1.0);
    const DATA_TYPE two  = SCALAR_VAL(2.0);

    DATA_TYPE k;
    DATA_TYPE a1, a2, a3, a4, a5, a6, a7, a8;
    DATA_TYPE b1, b2, c1, c2;

#pragma scop
   /* hoist expensive scalar computations and common subexpressions */
   const DATA_TYPE emalpha  = EXP_FUN(-alpha);
   const DATA_TYPE em2alpha = EXP_FUN(-two * alpha);
   const DATA_TYPE alpha_emalpha = alpha * emalpha;
   const DATA_TYPE one_minus_emalpha = one - emalpha;
   const DATA_TYPE denom = one + two * alpha_emalpha - EXP_FUN(two * alpha);

   k  = (one_minus_emalpha * one_minus_emalpha) / denom;
   a1 = k;
   a5 = k;
   a2 = k * emalpha * (alpha - one);
   a6 = a2;
   a3 = k * emalpha * (alpha + one);
   a7 = a3;
   a4 = -k * em2alpha;
   a8 = a4;
   b1 = POW_FUN(two, -alpha);
   b2 = -em2alpha;
   c1 = SCALAR_VAL(1.0);
   c2 = SCALAR_VAL(1.0);

   /* First horizontal causal pass: independent per row */
#pragma omp parallel for private(j, xm1, ym1, ym2) if (_PB_W * _PB_H > 1024)
   for (i = 0; i < _PB_W; i++) {
        ym1 = zero;
        ym2 = zero;
        xm1 = zero;
#pragma GCC ivdep
        for (j = 0; j < _PB_H; j++) {
            const DATA_TYPE in_ij = imgIn[i][j];
            const DATA_TYPE y = a1 * in_ij + a2 * xm1 + b1 * ym1 + b2 * ym2;
            y1[i][j] = y;
            xm1 = in_ij;
            ym2 = ym1;
            ym1 = y;
        }
    }

   /* First horizontal anti-causal pass: independent per row */
#pragma omp parallel for private(j, xp1, xp2, yp1, yp2) if (_PB_W * _PB_H > 1024)
   for (i = 0; i < _PB_W; i++) {
        yp1 = zero;
        yp2 = zero;
        xp1 = zero;
        xp2 = zero;
#pragma GCC ivdep
        for (j = _PB_H-1; j >= 0; j--) {
            const DATA_TYPE y = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
            y2[i][j] = y;
            xp2 = xp1;
            xp1 = imgIn[i][j];
            yp2 = yp1;
            yp1 = y;
        }
    }

   /* Combine results of first horizontal passes: independent per element */
#pragma omp parallel for private(j) if (_PB_W * _PB_H > 1024)
   for (i = 0; i < _PB_W; i++) {
#pragma GCC ivdep
        for (j = 0; j < _PB_H; j++) {
            imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
        }
   }

   /* Vertical causal pass: independent per column */
#pragma omp parallel for private(i, tm1, ym1, ym2) if (_PB_W * _PB_H > 1024)
    for (j = 0; j < _PB_H; j++) {
        tm1 = zero;
        ym1 = zero;
        ym2 = zero;
#pragma GCC ivdep
        for (i = 0; i < _PB_W; i++) {
            const DATA_TYPE out_ij = imgOut[i][j];
            const DATA_TYPE y = a5 * out_ij + a6 * tm1 + b1 * ym1 + b2 * ym2;
            y1[i][j] = y;
            tm1 = out_ij;
            ym2 = ym1;
            ym1 = y;
        }
    }

   /* Vertical anti-causal pass: independent per column */
#pragma omp parallel for private(i, tp1, tp2, yp1, yp2) if (_PB_W * _PB_H > 1024)
    for (j = 0; j < _PB_H; j++) {
        tp1 = zero;
        tp2 = zero;
        yp1 = zero;
        yp2 = zero;
#pragma GCC ivdep
        for (i = _PB_W-1; i >= 0; i--) {
            const DATA_TYPE y = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
            y2[i][j] = y;
            tp2 = tp1;
            tp1 = imgOut[i][j];
            yp2 = yp1;
            yp1 = y;
        }
    }

   /* Final combine: independent per element */
#pragma omp parallel for private(j) if (_PB_W * _PB_H > 1024)
    for (i = 0; i < _PB_W; i++) {
#pragma GCC ivdep
        for (j = 0; j < _PB_H; j++) {
            imgOut[i][j] = c2 * (y1[i][j] + y2[i][j]);
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
  int w = W;
  int h = H;

  /* Variable declaration/allocation. */
  DATA_TYPE alpha;
  POLYBENCH_2D_ARRAY_DECL(imgIn, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(imgOut, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y1, DATA_TYPE, W, H, w, h);
  POLYBENCH_2D_ARRAY_DECL(y2, DATA_TYPE, W, H, w, h);


  /* Initialize array(s). */
  init_array (w, h, &alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut));

  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Run kernel. */
  kernel_deriche (w, h, alpha, POLYBENCH_ARRAY(imgIn), POLYBENCH_ARRAY(imgOut), POLYBENCH_ARRAY(y1), POLYBENCH_ARRAY(y2));

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(w, h, POLYBENCH_ARRAY(imgOut)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(imgIn);
  POLYBENCH_FREE_ARRAY(imgOut);
  POLYBENCH_FREE_ARRAY(y1);
  POLYBENCH_FREE_ARRAY(y2);

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