#include <time.h>
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE],
         double real_twid[FFT_SIZE / 2], double img_twid[FFT_SIZE / 2]) {
    int even, odd, span, log_stage, rootindex;
    double temp_real, temp_img;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    log_stage = 0;
    for (span = FFT_SIZE >> 1; span; span >>= 1, log_stage++) {
        int span_mask = span;
        int log_shift = log_stage;
        for (odd = span; odd < FFT_SIZE; ) {
            odd |= span_mask;
            even = odd ^ span_mask;

            /* Butterfly on real part */
            temp_real   = real[even] + real[odd];
            real[odd]   = real[even] - real[odd];
            real[even]  = temp_real;

            /* Butterfly on imaginary part */
            temp_img    = img[even] + img[odd];
            img[odd]    = img[even] - img[odd];
            img[even]   = temp_img;

            rootindex = (even << log_shift) & (FFT_SIZE - 1);

            if (rootindex) {
                double r_odd = real[odd];
                double i_odd = img[odd];
                double tw_r  = real_twid[rootindex];
                double tw_i  = img_twid[rootindex];

                temp_real = tw_r * r_odd - tw_i * i_odd;
                temp_img  = tw_r * i_odd + tw_i * r_odd;

                real[odd] = temp_real;
                img[odd]  = temp_img;
            }

            ++odd;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc +=
        (kernel_end.tv_sec - kernel_start.tv_sec) +
        (kernel_end.tv_nsec - kernel_start.tv_nsec) * 1e-9;
}
