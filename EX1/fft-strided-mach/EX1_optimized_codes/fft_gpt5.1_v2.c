#include <time.h>
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]){
    int even, odd, span, log;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    log = 0;

    for (span = FFT_SIZE >> 1; span; span >>= 1, log++) {
        const int span_mask = span - 1;

        for (odd = span; odd < FFT_SIZE; odd++) {
            int rootindex;

            odd |= span;
            even = odd ^ span;

            {
                const double real_even = real[even];
                const double real_odd  = real[odd];
                const double img_even  = img[even];
                const double img_odd   = img[odd];

                const double real_sum  = real_even + real_odd;
                const double real_diff = real_even - real_odd;
                const double img_sum   = img_even + img_odd;
                const double img_diff  = img_even - img_odd;

                real[even] = real_sum;
                real[odd]  = real_diff;
                img[even]  = img_sum;
                img[odd]   = img_diff;
            }

            rootindex = (even << log) & (FFT_SIZE - 1);
            if (rootindex) {
                const double rt = real_twid[rootindex];
                const double it = img_twid[rootindex];
                const double r_odd = real[odd];
                const double i_odd = img[odd];

                const double t_real = rt * r_odd - it * i_odd;
                const double t_img  = rt * i_odd + it * r_odd;

                real[odd] = t_real;
                img[odd]  = t_img;
            }

            odd += span_mask;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
