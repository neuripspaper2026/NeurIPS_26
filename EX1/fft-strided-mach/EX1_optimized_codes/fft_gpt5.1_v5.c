#include <time.h>
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]){
    int even, odd, span, log;
    int rootindex;
    double temp;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    log = 0;
    for (span = FFT_SIZE >> 1; span; span >>= 1, ++log) {
        const int mask = FFT_SIZE - 1;
        int k = span;
        for (odd = span; odd < FFT_SIZE; ++odd) {
            k |= span;
            even = k ^ span;

            const double real_even = real[even];
            const double real_odd  = real[k];

            const double img_even  = img[even];
            const double img_odd   = img[k];

            const double real_sum  = real_even + real_odd;
            const double real_diff = real_even - real_odd;
            const double img_sum   = img_even + img_odd;
            const double img_diff  = img_even - img_odd;

            real[even] = real_sum;
            real[k]    = real_diff;
            img[even]  = img_sum;
            img[k]     = img_diff;

            rootindex = (even << log) & mask;
            if (rootindex) {
                const double twr = real_twid[rootindex];
                const double twi = img_twid[rootindex];

                const double r_odd = real[k];
                const double i_odd = img[k];

                const double r_temp = twr * r_odd - twi * i_odd;
                const double i_temp = twr * i_odd + twi * r_odd;

                real[k] = r_temp;
                img[k]  = i_temp;
            }
            ++k;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (double)(kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (double)(kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
