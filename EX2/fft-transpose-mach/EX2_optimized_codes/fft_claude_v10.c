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

<<<CODE>>>
#include "fft.h"
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static double fft_transpose_kernel_time_acc = 0.0;

void reset_fft_transpose_kernel_time(void) {
    fft_transpose_kernel_time_acc = 0.0;
}

double get_fft_transpose_kernel_time(void) {
    return fft_transpose_kernel_time_acc;
}

#define cmplx_M_x(a_x, a_y, b_x, b_y) ((a_x)*(b_x) - (a_y)*(b_y))
#define cmplx_M_y(a_x, a_y, b_x, b_y) ((a_x)*(b_y) + (a_y)*(b_x))
#define cmplx_MUL_x(a_x, a_y, b_x, b_y) ((a_x)*(b_x) - (a_y)*(b_y))
#define cmplx_MUL_y(a_x, a_y, b_x, b_y) ((a_x)*(b_y) + (a_y)*(b_x))
#define cmplx_mul_x(a_x, a_y, b_x, b_y) ((a_x)*(b_x) - (a_y)*(b_y))
#define cmplx_mul_y(a_x, a_y, b_x, b_y) ((a_x)*(b_y) + (a_y)*(b_x))
#define cmplx_add_x(a_x, b_x) ((a_x) + (b_x))
#define cmplx_add_y(a_y, b_y) ((a_y) + (b_y))
#define cmplx_sub_x(a_x, b_x) ((a_x) - (b_x))
#define cmplx_sub_y(a_y, b_y) ((a_y) - (b_y))
#define cm_fl_mul_x(a_x, b) ((b)*(a_x))
#define cm_fl_mul_y(a_y, b) ((b)*(a_y))

static inline void twiddles8(TYPE a_x[8], TYPE a_y[8], int i, int n) {
    static const int reversed8[8] = {0,4,2,6,1,5,3,7};
    TYPE phi, tmp, phi_x, phi_y;
    const TYPE factor = -2.0 * PI / n;
    
    for(int j = 1; j < 8; j++) {
        phi = factor * reversed8[j] * i;
        phi_x = cos(phi);
        phi_y = sin(phi);
        tmp = a_x[j];
        a_x[j] = cmplx_M_x(a_x[j], a_y[j], phi_x, phi_y);
        a_y[j] = cmplx_M_y(tmp, a_y[j], phi_x, phi_y);
    }
}

#define FF2(a0_x, a0_y, a1_x, a1_y) do { \
    TYPE c0_x = *(a0_x); \
    TYPE c0_y = *(a0_y); \
    *(a0_x) = cmplx_add_x(c0_x, *(a1_x)); \
    *(a0_y) = cmplx_add_y(c0_y, *(a1_y)); \
    *(a1_x) = cmplx_sub_x(c0_x, *(a1_x)); \
    *(a1_y) = cmplx_sub_y(c0_y, *(a1_y)); \
} while(0)

#define FFT4(a0_x, a0_y, a1_x, a1_y, a2_x, a2_y, a3_x, a3_y) do { \
    TYPE tmp; \
    FF2(a0_x, a0_y, a2_x, a2_y); \
    FF2(a1_x, a1_y, a3_x, a3_y); \
    tmp = *(a3_x); \
    *(a3_x) = *(a3_y); \
    *(a3_y) = -tmp; \
    FF2(a0_x, a0_y, a1_x, a1_y); \
    FF2(a2_x, a2_y, a3_x, a3_y); \
} while(0)

#define FFT8(a_x, a_y) do { \
    TYPE tmp_1; \
    FF2(&a_x[0], &a_y[0], &a_x[4], &a_y[4]); \
    FF2(&a_x[1], &a_y[1], &a_x[5], &a_y[5]); \
    FF2(&a_x[2], &a_y[2], &a_x[6], &a_y[6]); \
    FF2(&a_x[3], &a_y[3], &a_x[7], &a_y[7]); \
    tmp_1 = a_x[5]; \
    a_x[5] = M_SQRT1_2 * (a_x[5] + a_y[5]); \
    a_y[5] = M_SQRT1_2 * (tmp_1 - a_y[5]); \
    tmp_1 = a_x[6]; \
    a_x[6] = a_y[6]; \
    a_y[6] = -tmp_1; \
    tmp_1 = a_x[7]; \
    a_x[7] = M_SQRT1_2 * (-a_x[7] - a_y[7]); \
    a_y[7] = M_SQRT1_2 * (-tmp_1 + a_y[7]); \
    FFT4(&a_x[0], &a_y[0], &a_x[1], &a_y[1], &a_x[2], &a_y[2], &a_x[3], &a_y[3]); \
    FFT4(&a_x[4], &a_y[4], &a_x[5], &a_y[5], &a_x[6], &a_y[6], &a_x[7], &a_y[7]); \
} while(0)

static inline void loady8(TYPE a_y[8], const TYPE x[576], int offset, int sx) {
    a_y[0] = x[0*sx+offset];
    a_y[1] = x[1*sx+offset];
    a_y[2] = x[2*sx+offset];
    a_y[3] = x[3*sx+offset];
    a_y[4] = x[4*sx+offset];
    a_y[5] = x[5*sx+offset];
    a_y[6] = x[6*sx+offset];
    a_y[7] = x[7*sx+offset];
}

void fft1D_512(TYPE work_x[512], TYPE work_y[512]) {
    static const int reversed[8] = {0,4,2,6,1,5,3,7};
    const int stride = 64;
    
    TYPE DATA_x[512] __attribute__((aligned(64)));
    TYPE DATA_y[512] __attribute__((aligned(64)));
    TYPE smem[576] __attribute__((aligned(64)));
    
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    #pragma omp parallel
    {
        TYPE data_x[8] __attribute__((aligned(64)));
        TYPE data_y[8] __attribute__((aligned(64)));
        
        #pragma omp for nowait
        for(int tid = 0; tid < 64; tid++) {
            for(int i = 0; i < 8; i++) {
                data_x[i] = work_x[i*stride + tid];
                data_y[i] = work_y[i*stride + tid];
            }
            
            FFT8(data_x, data_y);
            twiddles8(data_x, data_y, tid, 512);
            
            int base = tid << 3;
            for(int i = 0; i < 8; i++) {
                DATA_x[base + i] = data_x[i];
                DATA_y[base + i] = data_y[i];
            }
        }
        
        #pragma omp barrier
        
        #pragma omp for nowait
        for(int tid = 0; tid < 64; tid++) {
            int hi = tid >> 3;
            int lo = tid & 7;
            int offset = (hi << 3) + lo;
            int base = tid << 3;
            
            smem[0*66 + offset] = DATA_x[base + 0];
            smem[4*66 + offset] = DATA_x[base + 1];
            smem[1*66 + offset] = DATA_x[base + 4];
            smem[5*66 + offset] = DATA_x[base + 5];
            smem[2*66 + offset] = DATA_x[base + 2];
            smem[6*66 + offset] = DATA_x[base + 3];
            smem[3*66 + offset] = DATA_x[base + 6];
            smem[7*66 + offset] = DATA_x[base + 7];
        }
        
        #pragma omp barrier
        
        #pragma omp for nowait
        for(int tid = 0; tid < 64; tid++) {
            int hi = tid >> 3;
            int lo = tid & 7;
            int offset = lo*66 + hi;
            int base = tid << 3;
            
            DATA_x[base + 0] = smem[0*8 + offset];
            DATA_x[base + 4] = smem[4*8 + offset];
            DATA_x[base + 1] = smem[1*8 + offset];
            DATA_x[base + 5] = smem[5*8 + offset];
            DATA_x[base + 2] = smem[2*8 + offset];
            DATA_x[base + 6] = smem[6*8 + offset];
            DATA_x[base + 3] = smem[3*8 + offset];
            DATA_x[base + 7] = smem[7*8 + offset];
        }
        
        #pragma omp barrier
        
        #pragma omp for nowait
        for(int tid = 0; tid < 64; tid++) {
            int hi = tid >> 3;
            int lo = tid & 7;
            int offset = (hi << 3) + lo;
            int base = tid << 3;
            
            smem[0*66 + offset] = DATA_y[base + 0];
            smem[4*66 + offset] = DATA_y[base + 1];
            smem[1*66 + offset] = DATA_y[base + 4];
            smem[5*66 + offset] = DATA_y[base + 5];
            smem[2*66 + offset] = DATA_y[base + 2];
            smem[6*66 + offset] = DATA_y[base + 3];
            smem[3*66 + offset] = DATA_y[base + 6];
            smem[7*66 + offset] = DATA_y[base + 7];
        }
        
        #pragma omp barrier
        
        #pragma omp for nowait
        for(int tid = 0; tid < 64; tid++) {
            int hi = tid >> 3;
            int lo = tid & 7;
            int offset = lo*66 + hi;
            int base = tid << 3;
            
            loady8(data_y, smem, offset, 8);
            
            for(int i = 0; i < 8; i++) {
                DATA_y[base + i] = data_y[i];
            }
        }
        
        #pragma omp barrier
        
        #pragma omp for nowait
        for(int tid = 0; tid < 64; tid++) {
            int base = tid << 3;
            
            for(int i = 0; i < 8; i++) {
                data_x[i] = DATA_x[base + i];
                data_y[i] = DATA_y[base + i];
            }
            
            FFT8(data_x, data_y);
            
            int hi = tid >> 3;
            twiddles8(data_x, data_y, hi, 64);
            
            for(int i = 0; i < 8; i++) {
                DATA_x[base + i] = data_x[i];
                DATA_y[base + i] = data_y[i];
            }
        }
        
        #pragma omp barrier
        
        #pragma omp for nowait
        for(int tid = 0; tid < 64; tid++) {
            int hi = tid >> 3;
            int lo = tid & 7;
            int offset = (hi << 3) + lo;
            int base = tid << 3;
            
            smem[0*72 + offset] = DATA_x[base + 0];
            smem[4*72 + offset] = DATA_x[base + 1];
            smem[1*72 + offset] = DATA_x[base + 4];
            smem[5*72 + offset] = DATA_x[base + 5];
            smem[2*72 + offset] = DATA_x[base + 2];
            smem[6*72 + offset] = DATA_x[base + 3];
            smem[3*72 + offset] = DATA_x[base + 6];
            smem[7*72 + offset] = DATA_x[base + 7];
        }
        
        #pragma omp barrier
        
        #pragma omp for nowait
        for(int tid = 0; tid < 64; tid++) {
            int hi = tid >> 3;
            int lo = tid & 7;
            int offset = hi*72 + lo;
            int base = tid << 3;
            
            DATA_x[base + 0] = smem[0*8 + offset];
            DATA_x[base + 4] = smem[4*8 + offset];
            DATA_x[base + 1] = smem[1*8 + offset];
            DATA_x[base + 5] = smem[5*8 + offset];
            DATA_x[base + 2] = smem[2*8 + offset];
            DATA_x[base + 6] = smem[6*8 + offset];
            DATA_x[base + 3] = smem[3*8 + offset];
            DATA_x[base + 7] = smem[7*8 + offset];
        }
        
        #pragma omp barrier
        
        #pragma omp for nowait
        for(int tid = 0; tid < 64; tid++) {
            int hi = tid >> 3;
            int lo = tid & 7;
            int offset = (hi << 3) + lo;
            int base = tid << 3;
            
            smem[0*72 + offset] = DATA_y[base + 0];
            smem[4*72 + offset] = DATA_y[base + 1];
            smem[1*72 + offset] = DATA_y[base + 4];
            smem[5*72 + offset]
