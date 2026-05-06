#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../fft.h"

static double fft_transpose_kernel_time_acc = 0.0;

void reset_fft_transpose_kernel_time(void) { fft_transpose_kernel_time_acc = 0.0; }
double get_fft_transpose_kernel_time(void) { return fft_transpose_kernel_time_acc; }

//////BEGIN TWIDDLES ////////
#define THREADS 64
#define cmplx_M_x(a_x, a_y, b_x, b_y) (a_x*b_x - a_y *b_y)
#define cmplx_M_y(a_x, a_y, b_x, b_y) (a_x*b_y + a_y *b_x)
#define cmplx_MUL_x(a_x, a_y, b_x, b_y ) (a_x*b_x - a_y*b_y)
#define cmplx_MUL_y(a_x, a_y, b_x, b_y ) (a_x*b_y + a_y*b_x)
#define cmplx_mul_x(a_x, a_y, b_x, b_y) (a_x*b_x - a_y*b_y)
#define cmplx_mul_y(a_x, a_y, b_x, b_y) (a_x*b_y + a_y*b_x)
#define cmplx_add_x(a_x, b_x) (a_x + b_x)
#define cmplx_add_y(a_y, b_y) (a_y + b_y)
#define cmplx_sub_x(a_x, b_x) (a_x - b_x)
#define cmplx_sub_y(a_y, b_y) (a_y - b_y)
#define cm_fl_mul_x(a_x, b) (b*a_x)
#define cm_fl_mul_y(a_y, b) (b*a_y)

void twiddles8(TYPE a_x[8], TYPE a_y[8], int i, int n){
    int reversed8[8] = {0,4,2,6,1,5,3,7};
    int j;
    TYPE phi, tmp, phi_x, phi_y;

    twiddles:for(j=1; j < 8; j++){
        phi = ((-2*PI*reversed8[j]/n)*i);
        phi_x = cos(phi);
        phi_y = sin(phi);
        tmp = a_x[j];
        a_x[j] = cmplx_M_x(a_x[j], a_y[j], phi_x, phi_y);
        a_y[j] = cmplx_M_y(tmp, a_y[j], phi_x, phi_y);
    }
}
////END TWIDDLES ////

#define FF2(a0_x, a0_y, a1_x, a1_y){			\
    TYPE c0_x = *a0_x;		\
    TYPE c0_y = *a0_y;		\
    *a0_x = cmplx_add_x(c0_x, *a1_x);	\
    *a0_y = cmplx_add_y(c0_y, *a1_y);	\
    *a1_x = cmplx_sub_x(c0_x, *a1_x);	\
    *a1_y = cmplx_sub_y(c0_y, *a1_y);	\
}

#define FFT4(a0_x, a0_y, a1_x, a1_y, a2_x, a2_y, a3_x, a3_y){           \
    TYPE exp_1_44_x;		\
    TYPE exp_1_44_y;		\
    TYPE tmp;			\
    exp_1_44_x =  0.0;		\
    exp_1_44_y =  -1.0;		\
    FF2( a0_x, a0_y, a2_x, a2_y);   \
    FF2( a1_x, a1_y, a3_x, a3_y);   \
    tmp = *a3_x;			\
    *a3_x = *a3_x*exp_1_44_x-*a3_y*exp_1_44_y;     	\
    *a3_y = tmp*exp_1_44_y - *a3_y*exp_1_44_x;    	\
    FF2( a0_x, a0_y, a1_x, a1_y );                  \
    FF2( a2_x, a2_y, a3_x, a3_y );                  \
}

#define FFT8(a_x, a_y)			\
{                                               \
    TYPE exp_1_8_x, exp_1_4_x, exp_3_8_x;	\
    TYPE exp_1_8_y, exp_1_4_y, exp_3_8_y;	\
    TYPE tmp_1;			\
    exp_1_8_x =  1;				\
    exp_1_8_y = -1;				\
    exp_1_4_x =  0;				\
    exp_1_4_y = -1;				\
    exp_3_8_x = -1;				\
    exp_3_8_y = -1;				\
    FF2( &a_x[0], &a_y[0], &a_x[4], &a_y[4]);			\
    FF2( &a_x[1], &a_y[1], &a_x[5], &a_y[5]);			\
    FF2( &a_x[2], &a_y[2], &a_x[6], &a_y[6]);			\
    FF2( &a_x[3], &a_y[3], &a_x[7], &a_y[7]);			\
    tmp_1 = a_x[5];							\
    a_x[5] = cm_fl_mul_x( cmplx_mul_x(a_x[5], a_y[5], exp_1_8_x, exp_1_8_y),  M_SQRT1_2 );	\
    a_y[5] = cm_fl_mul_y( cmplx_mul_y(tmp_1, a_y[5], exp_1_8_x, exp_1_8_y) , M_SQRT1_2 );	\
    tmp_1 = a_x[6];							\
    a_x[6] = cmplx_mul_x( a_x[6], a_y[6], exp_1_4_x , exp_1_4_y);	\
    a_y[6] = cmplx_mul_y( tmp_1, a_y[6], exp_1_4_x , exp_1_4_y);	\
    tmp_1 = a_x[7];							\
    a_x[7] = cm_fl_mul_x( cmplx_mul_x(a_x[7], a_y[7], exp_3_8_x, exp_3_8_y), M_SQRT1_2 );	\
    a_y[7] = cm_fl_mul_y( cmplx_mul_y(tmp_1, a_y[7], exp_3_8_x, exp_3_8_y) , M_SQRT1_2 );	\
    FFT4( &a_x[0], &a_y[0], &a_x[1], &a_y[1], &a_x[2], &a_y[2], &a_x[3], &a_y[3] );	\
    FFT4( &a_x[4], &a_y[4], &a_x[5], &a_y[5], &a_x[6], &a_y[6], &a_x[7], &a_y[7] );	\
}

void loadx8(TYPE a_x[], TYPE x[], int offset, int sx){
    a_x[0] = x[0*sx+offset];
    a_x[1] = x[1*sx+offset];
    a_x[2] = x[2*sx+offset];
    a_x[3] = x[3*sx+offset];
    a_x[4] = x[4*sx+offset];
    a_x[5] = x[5*sx+offset];
    a_x[6] = x[6*sx+offset];
    a_x[7] = x[7*sx+offset];
}

void loady8(TYPE a_y[], TYPE x[], int offset, int sx){
    a_y[0] = x[0*sx+offset];
    a_y[1] = x[1*sx+offset];
    a_y[2] = x[2*sx+offset];
    a_y[3] = x[3*sx+offset];
    a_y[4] = x[4*sx+offset];
    a_y[5] = x[5*sx+offset];
    a_y[6] = x[6*sx+offset];
    a_y[7] = x[7*sx+offset];
}

#include "fft.h"
#ifdef _OPENMP
#include <omp.h>
#endif

extern double fft_transpose_kernel_time_acc;

void fft1D_512(TYPE work_x[512], TYPE work_y[512]){
    int tid, hi, lo, stride;
    const int reversed[8] = {0,4,2,6,1,5,3,7};
    TYPE DATA_x[THREADS*8];
    TYPE DATA_y[THREADS*8];

    TYPE data_x[8];
    TYPE data_y[8];

    TYPE smem[8*8*9];

    stride = THREADS;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

#pragma omp parallel default(none) \
    shared(work_x, work_y, DATA_x, DATA_y, smem, stride, fft_transpose_kernel_time_acc) \
    firstprivate(reversed)
    {
        int tid_local, hi_local, lo_local, offset, sx_local;
        TYPE data_x_local[8];
        TYPE data_y_local[8];

        /* loop1: GLOBAL_LOAD + first FFT8 + first twiddle */
#pragma omp for schedule(static)
        for(tid_local = 0; tid_local < THREADS; tid_local++){
            const int base = tid_local;
            data_x_local[0] = work_x[0*stride + base];
            data_x_local[1] = work_x[1*stride + base];
            data_x_local[2] = work_x[2*stride + base];
            data_x_local[3] = work_x[3*stride + base];
            data_x_local[4] = work_x[4*stride + base];
            data_x_local[5] = work_x[5*stride + base];
            data_x_local[6] = work_x[6*stride + base];
            data_x_local[7] = work_x[7*stride + base];

            data_y_local[0] = work_y[0*stride + base];
            data_y_local[1] = work_y[1*stride + base];
            data_y_local[2] = work_y[2*stride + base];
            data_y_local[3] = work_y[3*stride + base];
            data_y_local[4] = work_y[4*stride + base];
            data_y_local[5] = work_y[5*stride + base];
            data_y_local[6] = work_y[6*stride + base];
            data_y_local[7] = work_y[7*stride + base];

            FFT8(data_x_local, data_y_local);
            twiddles8(data_x_local, data_y_local, tid_local, 512);

            const int dbase = tid_local * 8;
            DATA_x[dbase + 0] = data_x_local[0];
            DATA_x[dbase + 1] = data_x_local[1];
            DATA_x[dbase + 2] = data_x_local[2];
            DATA_x[dbase + 3] = data_x_local[3];
            DATA_x[dbase + 4] = data_x_local[4];
            DATA_x[dbase + 5] = data_x_local[5];
            DATA_x[dbase + 6] = data_x_local[6];
            DATA_x[dbase + 7] = data_x_local[7];

            DATA_y[dbase + 0] = data_y_local[0];
            DATA_y[dbase + 1] = data_y_local[1];
            DATA_y[dbase + 2] = data_y_local[2];
            DATA_y[dbase + 3] = data_y_local[3];
            DATA_y[dbase + 4] = data_y_local[4];
            DATA_y[dbase + 5] = data_y_local[5];
            DATA_y[dbase + 6] = data_y_local[6];
            DATA_y[dbase + 7] = data_y_local[7];
        }

        /* loop2 : transpose X into smem (stride 66) */
#pragma omp for schedule(static)
        for(tid_local = 0; tid_local < 64; tid_local++){
            hi_local = tid_local >> 3;
            lo_local = tid_local & 7;
            offset = hi_local * 8 + lo_local;
            sx_local = 66;

            const int dbase = tid_local * 8;
            smem[0*sx_local + offset] = DATA_x[dbase + 0];
            smem[4*sx_local + offset] = DATA_x[dbase + 1];
            smem[1*sx_local + offset] = DATA_x[dbase + 4];
            smem[5*sx_local + offset] = DATA_x[dbase + 5];
            smem[2*sx_local + offset] = DATA_x[dbase + 2];
            smem[6*sx_local + offset] = DATA_x[dbase + 3];
            smem[3*sx_local + offset] = DATA_x[dbase + 6];
            smem[7*sx_local + offset] = DATA_x[dbase + 7];
        }

        /* loop3 : read-back X from smem (stride 8) */
#pragma omp for schedule(static)
        for(tid_local = 0; tid_local < 64; tid_local++){
            hi_local = tid_local >> 3;
            lo_local = tid_local & 7;
            offset = lo_local * 66 + hi_local;
            sx_local = 8;

            const int dbase = tid_local * 8;
            DATA_x[dbase + 0] = smem[0*sx_local + offset];
            DATA_x[dbase + 4] = smem[4*sx_local + offset];
            DATA_x[dbase + 1] = smem[1*sx_local + offset];
            DATA_x[dbase + 5] = smem[5*sx_local + offset];
            DATA_x[dbase + 2] = smem[2*sx_local + offset];
            DATA_x[dbase + 6] = smem[6*sx_local + offset];
            DATA_x[dbase + 3] = smem[3*sx_local + offset];
            DATA_x[dbase + 7] = smem[7*sx_local + offset];
        }

        /* loop4 : transpose Y into smem (stride 66) */
#pragma omp for schedule(static)
        for(tid_local = 0; tid_local < 64; tid_local++){
            hi_local = tid_local >> 3;
            lo_local = tid_local & 7;
            offset = hi_local * 8 + lo_local;
            sx_local = 66;

            const int dbase = tid_local * 8;
            smem[0*sx_local + offset] = DATA_y[dbase + 0];
            smem[4*sx_local + offset] = DATA_y[dbase + 1];
            smem[1*sx_local + offset] = DATA_y[dbase + 4];
            smem[5*sx_local + offset] = DATA_y[dbase + 5];
            smem[2*sx_local + offset] = DATA_y[dbase + 2];
            smem[6*sx_local + offset] = DATA_y[dbase + 3];
            smem[3*sx_local + offset] = DATA_y[dbase + 6];
            smem[7*sx_local + offset] = DATA_y[dbase + 7];
        }

        /* loop5 : read-back Y via loady8 */
#pragma omp for schedule(static)
        for(tid_local = 0; tid_local < 64; tid_local++){
            hi_local = tid_local >> 3;
            lo_local = tid_local & 7;

            loady8(data_y_local, smem, lo_local * 66 + hi_local, 8);

            const int dbase = tid_local * 8;
            DATA_y[dbase + 0] = data_y_local[0];
            DATA_y[dbase + 1] = data_y_local[1];
            DATA_y[dbase + 2] = data_y_local[2];
            DATA_y[dbase + 3] = data_y_local[3];
            DATA_y[dbase + 4] = data_y_local[4];
            DATA_y[dbase + 5] = data_y_local[5];
            DATA_y[dbase + 6] = data_y_local[6];
            DATA_y[dbase + 7] = data_y_local[7];
        }

        /* loop6 : second FFT8 + second twiddle */
#pragma omp for schedule(static)
        for(tid_local = 0; tid_local < 64; tid_local++){
            const int dbase = tid_local * 8;
            data_x_local[0] = DATA_x[dbase + 0];
            data_x_local[1] = DATA_x[dbase + 1];
            data_x_local[2] = DATA_x[dbase + 2];
            data_x_local[3] = DATA_x[dbase + 3];
            data_x_local[4] = DATA_x[dbase + 4];
            data_x_local[5] = DATA_x[dbase + 5];
            data_x_local[6] = DATA_x[dbase + 6];
            data_x_local[7] = DATA_x[dbase + 7];

            data_y_local[0] = DATA_y[dbase + 0];
            data_y_local[1] = DATA_y[dbase + 1];
            data_y_local[2] = DATA_y[dbase + 2];
            data_y_local[3] = DATA_y[dbase + 3];
            data_y_local[4] = DATA_y[dbase + 4];
            data_y_local[5] = DATA_y[dbase + 5];
            data_y_local[6] = DATA_y[dbase + 6];
            data_y_local[7] = DATA_y[dbase + 7];

            FFT8(data_x_local, data_y_local);

            hi_local = tid_local >> 3;
            twiddles8(data_x_local, data_y_local, hi_local, 64);

            DATA_x[dbase + 0] = data_x_local[0];
            DATA_x[dbase + 1] = data_x_local[1];
            DATA_x[dbase + 2] = data_x_local[2];
            DATA_x[dbase + 3] = data_x_local[3];
            DATA_x[dbase + 4] = data_x_local[4];
            DATA_x[dbase + 5] = data_x_local[5];
            DATA_x[dbase + 6] = data_x_local[6];
            DATA_x[dbase + 7] = data_x_local[7];

            DATA_y[dbase + 0] = data_y_local[0];
            DATA_y[dbase + 1] = data_y_local[1];
            DATA_y[dbase + 2] = data_y_local[2];
            DATA_y[dbase + 3] = data_y_local[3];
            DATA_y[dbase + 4] = data_y_local[4];
            DATA_y[dbase + 5] = data_y_local[5];
            DATA_y[dbase + 6] = data_y_local[6];
            DATA_y[dbase + 7] = data_y_local[7];
        }

        /* loop7 : transpose X into smem (stride 72) */
#pragma omp for schedule(static)
        for(tid_local = 0; tid_local < 64; tid_local++){
            hi_local = tid_local >> 3;
            lo_local = tid_local & 7;
            offset = hi_local * 8 + lo_local;
            sx_local = 72;

            const int dbase = tid_local * 8;
            smem[0*sx_local + offset] = DATA_x[dbase + 0];
            smem[4*sx_local + offset] = DATA_x[dbase + 1];
            smem[1*sx_local + offset] = DATA_x[dbase + 4];
            smem[5*sx_local + offset] = DATA_x[dbase + 5];
            smem[2*sx_local + offset] = DATA_x[dbase + 2];
            smem[6*sx_local + offset] = DATA_x[dbase + 3];
            smem[3*sx_local + offset] = DATA_x[dbase + 6];
            smem[7*sx_local + offset] = DATA_x[dbase + 7];
        }

        /* loop8 : read-back X from smem (stride 8) */
#pragma omp for schedule(static)
        for(tid_local = 0; tid_local < 64; tid_local++){
            hi_local = tid_local >> 3;
            lo_local = tid_local & 7;
            offset = hi_local * 72 + lo_local;
            sx_local = 8;

            const int dbase = tid_local * 8;
            DATA_x[dbase + 0] = smem[0*sx_local + offset];
            DATA_x[dbase + 4] = smem[4*sx_local + offset];
            DATA_x[dbase + 1] = smem[1*sx_local + offset];
            DATA_x[dbase + 5] = smem[5*sx_local + offset];
            DATA_x[dbase + 2] = smem[2*sx_local + offset];
            DATA_x[dbase + 6] = smem[6*sx_local + offset];
            DATA_x[dbase + 3] = smem[3*sx_local + offset];
            DATA_x[dbase + 7] = smem[7*sx_local + offset];
        }

        /* loop9 : transpose Y into smem (stride 72) */
#pragma omp for schedule(static)
        for(tid_local = 0; tid_local < 64; tid_local++){
            hi_local = tid_local >> 3;
            lo_local = tid_local & 7;
            offset = hi_local * 8 + lo_local;
            sx_local = 72;

            const int dbase = tid_local * 8;
            smem[0*sx_local + offset] = DATA_y[dbase + 0];
            smem[4*sx_local + offset] = DATA_y[dbase + 1];
            smem[1*sx_local + offset] = DATA_y[dbase + 4];
            smem[5*sx_local + offset] = DATA_y[dbase + 5];
            smem[2*sx_local + offset] = DATA_y[dbase + 2];
            smem[6*sx_local + offset] = DATA_y[dbase + 3];
            smem[3*sx_local + offset] = DATA_y[dbase + 6];
            smem[7*sx_local + offset] = DATA_y[dbase + 7];
        }

        /* loop10 : read-back Y via loady8 */
#pragma omp for schedule(static)
        for(tid_local = 0; tid_local < 64; tid_local++){
            hi_local = tid_local >> 3;
            lo_local = tid_local & 7;

            loady8(data_y_local, smem, hi_local * 72 + lo_local, 8);

            const int dbase = tid_local * 8;
            DATA_y[dbase + 0] = data_y_local[0];
            DATA_y[dbase + 1] = data_y_local[1];
            DATA_y[dbase + 2] = data_y_local[2];
            DATA_y[dbase + 3] = data_y_local[3];
            DATA_y[dbase + 4] = data_y_local[4];
            DATA_y[dbase + 5] = data_y_local[5];
            DATA_y[dbase + 6] = data_y_local[6];
            DATA_y[dbase + 7] = data_y_local[7];
        }

        /* loop11 : final FFT8 and global store */
#pragma omp for schedule(static)
        for(tid_local = 0; tid_local < 64; tid_local++){
            const int dbase = tid_local * 8;
            data_y_local[0] = DATA_y[dbase + 0];
            data_y_local[1] = DATA_y[dbase + 1];
            data_y_local[2] = DATA_y[dbase + 2];
            data_y_local[3] = DATA_y[dbase + 3];
            data_y_local[4] = DATA_y[dbase + 4];
            data_y_local[5] = DATA_y[dbase + 5];
            data_y_local[6] = DATA_y[dbase + 6];
            data_y_local[7] = DATA_y[dbase + 7];

            data_x_local[0] = DATA_x[dbase + 0];
            data_x_local[1] = DATA_x[dbase + 1];
            data_x_local[2] = DATA_x[dbase + 2];
            data_x_local[3] = DATA_x[dbase + 3];
            data_x_local[4] = DATA_x[dbase + 4];
            data_x_local[5] = DATA_x[dbase + 5];
            data_x_local[6] = DATA_x[dbase + 6];
            data_x_local[7] = DATA_x[dbase + 7];

            FFT8(data_x_local, data_y_local);

            work_x[0*stride + tid_local] = data_x_local[reversed[0]];
            work_x[1*stride + tid_local] = data_x_local[reversed[1]];
            work_x[2*stride + tid_local] = data_x_local[reversed[2]];
            work_x[3*stride + tid_local] = data_x_local[reversed[3]];
            work_x[4*stride + tid_local] = data_x_local[reversed[4]];
            work_x[5*stride + tid_local] = data_x_local[reversed[5]];
            work_x[6*stride + tid_local] = data_x_local[reversed[6]];
            work_x[7*stride + tid_local] = data_x_local[reversed[7]];

            work_y[0*stride + tid_local] = data_y_local[reversed[0]];
            work_y[1*stride + tid_local] = data_y_local[reversed[1]];
            work_y[2*stride + tid_local] = data_y_local[reversed[2]];
            work_y[3*stride + tid_local] = data_y_local[reversed[3]];
            work_y[4*stride + tid_local] = data_y_local[reversed[4]];
            work_y[5*stride + tid_local] = data_y_local[reversed[5]];
            work_y[6*stride + tid_local] = data_y_local[reversed[6]];
            work_y[7*stride + tid_local] = data_y_local[reversed[7]];
        }
    } /* end parallel */

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_transpose_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                     (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
