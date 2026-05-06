#include <time.h>
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]){
    int even, odd, span, log;
    double temp;
    log = 0;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    outer: for (span = FFT_SIZE >> 1; span; span >>= 1, log++) {
        inner: for (odd = span; odd < FFT_SIZE; ) {
            even = odd ^ span;

            // Cooley-Tukey butterflies
            double real_even = real[even];
            double real_odd  = real[odd];
            double img_even  = img[even];
            double img_odd   = img[odd];

            double real_sum  = real_even + real_odd;
            double real_diff = real_even - real_odd;
            double img_sum   = img_even + img_odd;
            double img_diff  = img_even - img_odd;

            real[even] = real_sum;
            real[odd]  = real_diff;
            img[even]  = img_sum;
            img[odd]   = img_diff;

            int rootindex = (even << log) & (FFT_SIZE - 1);
            if (rootindex) {
                double rt = real_twid[rootindex];
                double it = img_twid[rootindex];

                double real_odd_val = real_diff;
                double img_odd_val  = img_diff;

                double tmp_real = rt * real_odd_val - it * img_odd_val;
                double tmp_img  = rt * img_odd_val  + it * real_odd_val;

                real[odd] = tmp_real;
                img[odd]  = tmp_img;
            }

            odd = (odd + 1) | span;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
