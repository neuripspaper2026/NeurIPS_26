#include <time.h>
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

<<<CODE>>>
#include "fft.h"
#include <time.h>

static double fft_transpose_kernel_time_acc = 0.0;

void reset_fft_transpose_kernel_time(void) {
    fft_transpose_kernel_time_acc = 0.0;
}

double get_fft_transpose_kernel_time(void) {
    return fft_transpose_kernel_time_acc;
}

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

static inline void twiddles8(TYPE a_x[8], TYPE a_y[8], int i, int n){
    int reversed8[8] = {0,4,2,6,1,5,3,7};
    int j;
    TYPE phi, tmp, phi_x, phi_y;

    for(j=1; j < 8; j++){
        phi = ((-2*PI*reversed8[j]/n)*i);
        phi_x = cos(phi);
        phi_y = sin(phi);
        tmp = a_x[j];
        a_x[j] = cmplx_M_x(a_x[j], a_y[j], phi_x, phi_y);
        a_y[j] = cmplx_M_y(tmp, a_y[j], phi_x, phi_y);
    }
}

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

static inline void loadx8(TYPE a_x[], TYPE x[], int offset, int sx){
    a_x[0] = x[0*sx+offset];
    a_x[1] = x[1*sx+offset];
    a_x[2] = x[2*sx+offset];
    a_x[3] = x[3*sx+offset];
    a_x[4] = x[4*sx+offset];
    a_x[5] = x[5*sx+offset];
    a_x[6] = x[6*sx+offset];
    a_x[7] = x[7*sx+offset];
}

static inline void loady8(TYPE a_y[], TYPE x[], int offset, int sx){
    a_y[0] = x[0*sx+offset];
    a_y[1] = x[1*sx+offset];
    a_y[2] = x[2*sx+offset];
    a_y[3] = x[3*sx+offset];
    a_y[4] = x[4*sx+offset];
    a_y[5] = x[5*sx+offset];
    a_y[6] = x[6*sx+offset];
    a_y[7] = x[7*sx+offset];
}

void fft1D_512(TYPE work_x[512], TYPE work_y[512]){
    int tid, hi, lo;
    const int stride = 64;
    const int reversed[] = {0,4,2,6,1,5,3,7};
    TYPE DATA_x[512] __attribute__((aligned(64)));
    TYPE DATA_y[512] __attribute__((aligned(64)));
    TYPE smem[576] __attribute__((aligned(64)));

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for(tid = 0; tid < 64; tid++){
        TYPE data_x[8];
        TYPE data_y[8];
        
        int base = tid;
        data_x[0] = work_x[base];
        data_x[1] = work_x[base + 64];
        data_x[2] = work_x[base + 128];
        data_x[3] = work_x[base + 192];
        data_x[4] = work_x[base + 256];
        data_x[5] = work_x[base + 320];
        data_x[6] = work_x[base + 384];
        data_x[7] = work_x[base + 448];

        data_y[0] = work_y[base];
        data_y[1] = work_y[base + 64];
        data_y[2] = work_y[base + 128];
        data_y[3] = work_y[base + 192];
        data_y[4] = work_y[base + 256];
        data_y[5] = work_y[base + 320];
        data_y[6] = work_y[base + 384];
        data_y[7] = work_y[base + 448];

        FFT8(data_x, data_y);
        twiddles8(data_x, data_y, tid, 512);

        int tid8 = tid << 3;
        DATA_x[tid8]     = data_x[0];
        DATA_x[tid8 + 1] = data_x[1];
        DATA_x[tid8 + 2] = data_x[2];
        DATA_x[tid8 + 3] = data_x[3];
        DATA_x[tid8 + 4] = data_x[4];
        DATA_x[tid8 + 5] = data_x[5];
        DATA_x[tid8 + 6] = data_x[6];
        DATA_x[tid8 + 7] = data_x[7];

        DATA_y[tid8]     = data_y[0];
        DATA_y[tid8 + 1] = data_y[1];
        DATA_y[tid8 + 2] = data_y[2];
        DATA_y[tid8 + 3] = data_y[3];
        DATA_y[tid8 + 4] = data_y[4];
        DATA_y[tid8 + 5] = data_y[5];
        DATA_y[tid8 + 6] = data_y[6];
        DATA_y[tid8 + 7] = data_y[7];
    }

    for(tid = 0; tid < 64; tid++){
        hi = tid >> 3;
        lo = tid & 7;
        int offset = (hi << 3) + lo;
        int tid8 = tid << 3;
        smem[offset] = DATA_x[tid8];
        smem[264 + offset] = DATA_x[tid8 + 1];
        smem[66 + offset] = DATA_x[tid8 + 4];
        smem[330 + offset] = DATA_x[tid8 + 5];
        smem[132 + offset] = DATA_x[tid8 + 2];
        smem[396 + offset] = DATA_x[tid8 + 3];
        smem[198 + offset] = DATA_x[tid8 + 6];
        smem[462 + offset] = DATA_x[tid8 + 7];
    }

    for(tid = 0; tid < 64; tid++){
        hi = tid >> 3;
        lo = tid & 7;
        int offset = lo * 66 + hi;
        int tid8 = tid << 3;
        DATA_x[tid8] = smem[offset];
        DATA_x[tid8 + 4] = smem[offset + 32];
        DATA_x[tid8 + 1] = smem[offset + 8];
        DATA_x[tid8 + 5] = smem[offset + 40];
        DATA_x[tid8 + 2] = smem[offset + 16];
        DATA_x[tid8 + 6] = smem[offset + 48];
        DATA_x[tid8 + 3] = smem[offset + 24];
        DATA_x[tid8 + 7] = smem[offset + 56];
    }

    for(tid = 0; tid < 64; tid++){
        hi = tid >> 3;
        lo = tid & 7;
        int offset = (hi << 3) + lo;
        int tid8 = tid << 3;
        smem[offset] = DATA_y[tid8];
        smem[264 + offset] = DATA_y[tid8 + 1];
        smem[66 + offset] = DATA_y[tid8 + 4];
        smem[330 + offset] = DATA_y[tid8 + 5];
        smem[132 + offset] = DATA_y[tid8 + 2];
        smem[396 + offset] = DATA_y[tid8 + 3];
        smem[198 + offset] = DATA_y[tid8 + 6];
        smem[462 + offset] = DATA_y[tid8 + 7];
    }

    for(tid = 0; tid < 64; tid++){
        hi = tid >> 3;
        lo = tid & 7;
        int offset = lo * 66 + hi;
        int tid8 = tid << 3;
        DATA_y[tid8] = smem[offset];
        DATA_y[tid8 + 1] = smem[offset + 8];
        DATA_y[tid8 + 2] = smem[offset + 16];
        DATA_y[tid8 + 3] = smem[offset + 24];
        DATA_y[tid8 + 4] = smem[offset
