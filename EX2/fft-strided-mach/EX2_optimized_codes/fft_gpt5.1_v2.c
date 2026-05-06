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
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    log = 0;

    outer:
    for (span = FFT_SIZE >> 1; span; span >>= 1, log++) {
        int mask = ~span;
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int idx = span; idx < FFT_SIZE; idx++) {
            int even = idx & mask;
            int odd  = even | span;

            double real_even = real[even];
            double real_odd  = real[odd];
            double img_even  = img[even];
            double img_odd   = img[odd];

            double temp_real = real_even + real_odd;
            double diff_real = real_even - real_odd;
            double temp_img  = img_even + img_odd;
            double diff_img  = img_even - img_odd;

            real[even] = temp_real;
            real[odd]  = diff_real;
            img[even]  = temp_img;
            img[odd]   = diff_img;

            int rootindex = (even << log) & (FFT_SIZE - 1);
            if (rootindex) {
                double rt = real_twid[rootindex];
                double it = img_twid[rootindex];

                double odd_real = real[odd];
                double odd_img  = img[odd];

                double tw_real = rt * odd_real - it * odd_img;
                double tw_img  = rt * odd_img + it * odd_real;

                real[odd] = tw_real;
                img[odd]  = tw_img;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
