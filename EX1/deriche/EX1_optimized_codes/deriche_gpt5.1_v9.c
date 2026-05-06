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
        const DATA_TYPE mone = SCALAR_VAL(-1.0);
        const DATA_TYPE mtwo = SCALAR_VAL(-2.0);

        const DATA_TYPE e_malpha  = EXP_FUN(mone * alpha);
        const DATA_TYPE e_m2alpha = EXP_FUN(mtwo * alpha);

        const DATA_TYPE num  = (one - e_malpha) * (one - e_malpha);
        const DATA_TYPE den  = one + two * alpha * e_malpha - EXP_FUN(two * alpha);
        k  = num / den;

        a1 = k;
        a5 = k;

        const DATA_TYPE tmp_a26 = k * e_malpha;
        a2 = tmp_a26 * (alpha - one);
        a6 = tmp_a26 * (alpha - one);

        const DATA_TYPE tmp_a37 = k * e_malpha;
        a3 = tmp_a37 * (alpha + one);
        a7 = tmp_a37 * (alpha + one);

        const DATA_TYPE tmp_a48 = -k * e_m2alpha;
        a4 = tmp_a48;
        a8 = tmp_a48;

        b1 = POW_FUN(two, -alpha);
        b2 = -e_m2alpha;

        c1 = SCALAR_VAL(1.0);
        c2 = SCALAR_VAL(1.0);
    }

    for (i = 0; i < _PB_W; i++) {
        DATA_TYPE ym1_l = SCALAR_VAL(0.0);
        DATA_TYPE ym2_l = SCALAR_VAL(0.0);
        DATA_TYPE xm1_l = SCALAR_VAL(0.0);
        for (j = 0; j < _PB_H; j++) {
            const DATA_TYPE in_ij = imgIn[i][j];
            const DATA_TYPE y1_ij = a1 * in_ij + a2 * xm1_l + b1 * ym1_l + b2 * ym2_l;
            y1[i][j] = y1_ij;
            xm1_l = in_ij;
            ym2_l = ym1_l;
            ym1_l = y1_ij;
        }
    }

    for (i = 0; i < _PB_W; i++) {
        DATA_TYPE yp1_l = SCALAR_VAL(0.0);
        DATA_TYPE yp2_l = SCALAR_VAL(0.0);
        DATA_TYPE xp1_l = SCALAR_VAL(0.0);
        DATA_TYPE xp2_l = SCALAR_VAL(0.0);
        for (j = _PB_H - 1; j >= 0; j--) {
            const DATA_TYPE y2_ij = a3 * xp1_l + a4 * xp2_l + b1 * yp1_l + b2 * yp2_l;
            y2[i][j] = y2_ij;
            xp2_l = xp1_l;
            xp1_l = imgIn[i][j];
            yp2_l = yp1_l;
            yp1_l = y2_ij;
        }
    }

    for (i = 0; i < _PB_W; i++) {
        for (j = 0; j < _PB_H; j++) {
            imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
        }
    }

    for (j = 0; j < _PB_H; j++) {
        DATA_TYPE tm1_l = SCALAR_VAL(0.0);
        DATA_TYPE ym1_l = SCALAR_VAL(0.0);
        DATA_TYPE ym2_l = SCALAR_VAL(0.0);
        for (i = 0; i < _PB_W; i++) {
            const DATA_TYPE out_ij = imgOut[i][j];
            const DATA_TYPE y1_ij = a5 * out_ij + a6 * tm1_l + b1 * ym1_l + b2 * ym2_l;
            y1[i][j] = y1_ij;
            tm1_l = out_ij;
            ym2_l = ym1_l;
            ym1_l = y1_ij;
        }
    }

    for (j = 0; j < _PB_H; j++) {
        DATA_TYPE tp1_l = SCALAR_VAL(0.0);
        DATA_TYPE tp2_l = SCALAR_VAL(0.0);
        DATA_TYPE yp1_l = SCALAR_VAL(0.0);
        DATA_TYPE yp2_l = SCALAR_VAL(0.0);
        for (i = _PB_W - 1; i >= 0; i--) {
            const DATA_TYPE y2_ij = a7 * tp1_l + a8 * tp2_l + b1 * yp1_l + b2 * yp2_l;
            y2[i][j] = y2_ij;
            tp2_l = tp1_l;
            tp1_l = imgOut[i][j];
            yp2_l = yp1_l;
            yp1_l = y2_ij;
        }
    }

    for (i = 0; i < _PB_W; i++) {
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