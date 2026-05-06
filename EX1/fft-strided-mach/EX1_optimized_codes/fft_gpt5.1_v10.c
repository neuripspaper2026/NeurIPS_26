#include <time.h>
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]){
    int even, odd, span, log, rootindex;
    double temp;
    log = 0;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (span = FFT_SIZE >> 1; span; span >>= 1, log++) {
        int const mask = span;
        int const size_mask = FFT_SIZE - 1;

        for (odd = span; odd < FFT_SIZE; odd++) {
            int idx = odd | mask;
            even = idx ^ mask;

            double real_even = real[even];
            double real_odd  = real[idx];
            double img_even  = img[even];
            double img_odd   = img[idx];

            temp        = real_even + real_odd;
            real_odd    = real_even - real_odd;
            real_even   = temp;

            temp        = img_even + img_odd;
            img_odd     = img_even - img_odd;
            img_even    = temp;

            real[even] = real_even;
            real[idx]  = real_odd;
            img[even]  = img_even;
            img[idx]   = img_odd;

            rootindex = (even << log) & size_mask;
            if (rootindex) {
                double twr = real_twid[rootindex];
                double twi = img_twid[rootindex];

                double r_odd = real_odd;
                double i_odd = img_odd;

                temp     = twr * r_odd - twi * i_odd;
                img[idx] = twr * i_odd + twi * r_odd;
                real[idx]= temp;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
