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
   {
     const DATA_TYPE one  = SCALAR_VAL(1.0);
     const DATA_TYPE two  = SCALAR_VAL(2.0);
     const DATA_TYPE zero = SCALAR_VAL(0.0);
     const DATA_TYPE neg_one = SCALAR_VAL(-1.0);
     const DATA_TYPE neg_two = SCALAR_VAL(-2.0);

     const DATA_TYPE exp_neg_alpha   = EXP_FUN(-alpha);
     const DATA_TYPE exp_neg_2alpha  = EXP_FUN(neg_two * alpha);
     const DATA_TYPE two_alpha       = two * alpha;

     const DATA_TYPE num  = (one - exp_neg_alpha) * (one - exp_neg_alpha);
     const DATA_TYPE den  = one + two_alpha * exp_neg_alpha - EXP_FUN(two_alpha);
     k  = num / den;

     const DATA_TYPE km_exp_neg_alpha = k * exp_neg_alpha;

     a1 = k;
     a5 = k;
     a2 = km_exp_neg_alpha * (alpha - one);
     a6 = a2;
     a3 = km_exp_neg_alpha * (alpha + one);
     a7 = a3;
     a4 = -k * exp_neg_2alpha;
     a8 = a4;

     b1 =  POW_FUN(two, neg_one);
     b2 = -exp_neg_2alpha;
     c1 = one;
     c2 = one;

     const int w_bound = _PB_W;
     const int h_bound = _PB_H;

     for (i = 0; i < w_bound; i++) {
        ym1 = zero;
        ym2 = zero;
        xm1 = zero;
        for (j = 0; j < h_bound; j++) {
            const DATA_TYPE in_ij = imgIn[i][j];
            const DATA_TYPE y_val = a1 * in_ij + a2 * xm1 + b1 * ym1 + b2 * ym2;
            y1[i][j] = y_val;
            xm1 = in_ij;
            ym2 = ym1;
            ym1 = y_val;
        }
     }

     for (i = 0; i < w_bound; i++) {
        yp1 = zero;
        yp2 = zero;
        xp1 = zero;
        xp2 = zero;
        for (j = h_bound - 1; j >= 0; j--) {
            const DATA_TYPE y_val = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
            y2[i][j] = y_val;
            xp2 = xp1;
            xp1 = imgIn[i][j];
            yp2 = yp1;
            yp1 = y_val;
        }
     }

     for (i = 0; i < w_bound; i++) {
        for (j = 0; j < h_bound; j++) {
            imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
        }
     }

     for (j = 0; j < h_bound; j++) {
        tm1 = zero;
        ym1 = zero;
        ym2 = zero;
        for (i = 0; i < w_bound; i++) {
            const DATA_TYPE out_ij = imgOut[i][j];
            const DATA_TYPE y_val  = a5 * out_ij + a6 * tm1 + b1 * ym1 + b2 * ym2;
            y1[i][j] = y_val;
            tm1 = out_ij;
            ym2 = ym1;
            ym1 = y_val;
        }
     }

     for (j = 0; j < h_bound; j++) {
        tp1 = zero;
        tp2 = zero;
        yp1 = zero;
        yp2 = zero;
        for (i = w_bound - 1; i >= 0; i--) {
            const DATA_TYPE y_val = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
            y2[i][j] = y_val;
            tp2 = tp1;
            tp1 = imgOut[i][j];
            yp2 = yp1;
            yp1 = y_val;
        }
     }

     for (i = 0; i < w_bound; i++) {
        for (j = 0; j < h_bound; j++) {
            imgOut[i][j] = c2 * (y1[i][j] + y2[i][j]);
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