#include <time.h>
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE],
         double real_twid[FFT_SIZE / 2], double img_twid[FFT_SIZE / 2]) {
    int span, log = 0;

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (span = FFT_SIZE >> 1; span; span >>= 1, ++log) {
        int odd;
        int span_mask = span;
        int span_xor_mask = span;
        int rootindex_mask = FFT_SIZE - 1;
        for (odd = span; odd < FFT_SIZE; ++odd) {
            int odd_index = odd | span_mask;
            int even_index = odd_index ^ span_xor_mask;

            double real_even = real[even_index];
            double real_odd  = real[odd_index];

            double img_even  = img[even_index];
            double img_odd   = img[odd_index];

            double real_sum  = real_even + real_odd;
            double real_diff = real_even - real_odd;
            real[even_index] = real_sum;
            real[odd_index]  = real_diff;

            double img_sum   = img_even + img_odd;
            double img_diff  = img_even - img_odd;
            img[even_index]  = img_sum;
            img[odd_index]   = img_diff;

            int rootindex = (even_index << log) & rootindex_mask;
            if (rootindex) {
                double tw_re = real_twid[rootindex];
                double tw_im = img_twid[rootindex];

                double x_re = real[odd_index];
                double x_im = img[odd_index];

                double tmp_re = tw_re * x_re - tw_im * x_im;
                double tmp_im = tw_re * x_im + tw_im * x_re;

                real[odd_index] = tmp_re;
                img[odd_index]  = tmp_im;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc +=
        (kernel_end.tv_sec - kernel_start.tv_sec) +
        (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
