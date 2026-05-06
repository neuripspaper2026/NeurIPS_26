#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]){
    int span, log;
    log = 0;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for(span = FFT_SIZE >> 1; span; span >>= 1, log++){
        int odd;
        /* Hoist frequently used constants */
        const int fft_mask = FFT_SIZE - 1;
        const int cur_span = span;
        const int cur_log  = log;

        /* Parallelize outer "odd" loop; each iteration writes unique indices
           (even, odd), so no data races occur. */
        #ifdef _OPENMP
        #pragma omp parallel for private(odd) schedule(static)
        #endif
        for(odd = cur_span; odd < FFT_SIZE; ++odd){
            int idx_odd = odd | cur_span;
            int idx_even = idx_odd ^ cur_span;

            double real_even = real[idx_even];
            double real_odd  = real[idx_odd];
            double img_even  = img[idx_even];
            double img_odd   = img[idx_odd];

            /* Butterfly for real part */
            double temp_real = real_even + real_odd;
            real_odd = real_even - real_odd;
            real_even = temp_real;

            /* Butterfly for imaginary part */
            double temp_img = img_even + img_odd;
            img_odd = img_even - img_odd;
            img_even = temp_img;

            real[idx_even] = real_even;
            img[idx_even]  = img_even;

            /* Twiddle factor application */
            int rootindex = (idx_even << cur_log) & fft_mask;
            if(rootindex){
                double tw_r = real_twid[rootindex];
                double tw_i = img_twid[rootindex];

                double r_odd = real_odd;
                double i_odd = img_odd;

                double new_real_odd = tw_r * r_odd - tw_i * i_odd;
                double new_img_odd  = tw_r * i_odd + tw_i * r_odd;

                real_odd = new_real_odd;
                img_odd  = new_img_odd;
            }

            real[idx_odd] = real_odd;
            img[idx_odd]  = img_odd;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
