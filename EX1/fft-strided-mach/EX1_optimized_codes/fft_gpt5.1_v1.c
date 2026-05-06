#include <time.h>
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE],
         double real_twid[FFT_SIZE / 2], double img_twid[FFT_SIZE / 2]) {
    int span, log;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    log = 0;
    for (span = FFT_SIZE >> 1; span; span >>= 1, log++) {
        int odd;
        for (odd = span; odd < FFT_SIZE; odd++) {
            int idx = odd | span;
            int even = idx ^ span;
            int rootindex;

            double real_even = real[even];
            double real_odd  = real[idx];
            double img_even  = img[even];
            double img_odd   = img[idx];

            double tmp_real = real_even + real_odd;
            real_odd        = real_even - real_odd;
            real_even       = tmp_real;

            double tmp_img = img_even + img_odd;
            img_odd        = img_even - img_odd;
            img_even       = tmp_img;

            real[even] = real_even;
            img[even]  = img_even;

            rootindex = (even << log) & (FFT_SIZE - 1);
            if (rootindex) {
                double twr = real_twid[rootindex];
                double twi = img_twid[rootindex];

                double t_real = twr * real_odd - twi * img_odd;
                double t_img  = twr * img_odd + twi * real_odd;

                real[idx] = t_real;
                img[idx]  = t_img;
            } else {
                real[idx] = real_odd;
                img[idx]  = img_odd;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc +=
        (kernel_end.tv_sec - kernel_start.tv_sec) +
        (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
