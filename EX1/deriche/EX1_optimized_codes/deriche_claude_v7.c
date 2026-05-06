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

    DATA_TYPE k;
    DATA_TYPE a1, a2, a3, a4, a5, a6, a7, a8;
    DATA_TYPE b1, b2, c1, c2;

#pragma scop
   DATA_TYPE exp_alpha = EXP_FUN(-alpha);
   DATA_TYPE exp_2alpha = EXP_FUN(SCALAR_VAL(-2.0)*alpha);
   DATA_TYPE one_minus_exp = SCALAR_VAL(1.0) - exp_alpha;
   k = one_minus_exp * one_minus_exp / (SCALAR_VAL(1.0) + SCALAR_VAL(2.0)*alpha*exp_alpha - exp_2alpha);
   a1 = a5 = k;
   DATA_TYPE k_exp_alpha = k * exp_alpha;
   a2 = a6 = k_exp_alpha * (alpha - SCALAR_VAL(1.0));
   a3 = a7 = k_exp_alpha * (alpha + SCALAR_VAL(1.0));
   a4 = a8 = -k * exp_2alpha;
   b1 = POW_FUN(SCALAR_VAL(2.0), -alpha);
   b2 = -exp_2alpha;
   c1 = c2 = 1;

   for (i=0; i<_PB_W; i++) {
        ym1 = SCALAR_VAL(0.0);
        ym2 = SCALAR_VAL(0.0);
        xm1 = SCALAR_VAL(0.0);
        for (j=0; j<_PB_H; j++) {
            DATA_TYPE imgIn_ij = imgIn[i][j];
            y1[i][j] = a1*imgIn_ij + a2*xm1 + b1*ym1 + b2*ym2;
            xm1 = imgIn_ij;
            ym2 = ym1;
            ym1 = y1[i][j];
        }
    }

    for (i=0; i<_PB_W; i++) {
        yp1 = SCALAR_VAL(0.0);
        yp2 = SCALAR_VAL(0.0);
        xp1 = SCALAR_VAL(0.0);
        xp2 = SCALAR_VAL(0.0);
        for (j=_PB_H-1; j>=0; j--) {
            y2[i][j] = a3*xp1 + a4*xp2 + b1*yp1 + b2*yp2;
            xp2 = xp1;
            xp1 = imgIn[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for (i=0; i<_PB_W; i++) {
        DATA_TYPE *y1_i = y1[i];
        DATA_TYPE *y2_i = y2[i];
        DATA_TYPE *imgOut_i = imgOut[i];
        for (j=0; j<_PB_H; j++) {
            imgOut_i[j] = c1 * (y1_i[j] + y2_i[j]);
        }
    }

    for (j=0; j<_PB_H; j++) {
        tm1 = SCALAR_VAL(0.0);
        ym1 = SCALAR_VAL(0.0);
        ym2 = SCALAR_VAL(0.0);
        for (i=0; i<_PB_W; i++) {
            DATA_TYPE imgOut_ij = imgOut[i][j];
            y1[i][j] = a5*imgOut_ij + a6*tm1 + b1*ym1 + b2*ym2;
            tm1 = imgOut_ij;
            ym2 = ym1;
            ym1 = y1[i][j];
        }
    }

    for (j=0; j<_PB_H; j++) {
        tp1 = SCALAR_VAL(0.0);
        tp2 = SCALAR_VAL(0.0);
        yp1 = SCALAR_VAL(0.0);
        yp2 = SCALAR_VAL(0.0);
        for (i=_PB_W-1; i>=0; i--) {
            y2[i][j] = a7*tp1 + a8*tp2 + b1*yp1 + b2*yp2;
            tp2 = tp1;
            tp1 = imgOut[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    for (i=0; i<_PB_W; i++) {
        DATA_TYPE *y1_i = y1[i];
        DATA_TYPE *y2_i = y2[i];
        DATA_TYPE *imgOut_i = imgOut[i];
        for (j=0; j<_PB_H; j++)
            imgOut_i[j] = c2*(y1_i[j] + y2_i[j]);
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