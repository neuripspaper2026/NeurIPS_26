#include <time.h>
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]){
    int even, odd, span, log;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (span = FFT_SIZE >> 1, log = 0; span; span >>= 1, log++) {
        const int bitmask = FFT_SIZE - 1;
        for (odd = span; odd < FFT_SIZE; odd++) {
            double r_even, r_odd, i_even, i_odd;
            double r_tmp, i_tmp;
            int rootindex;

            odd |= span;
            even = odd ^ span;

            r_even = real[even];
            r_odd  = real[odd];
            i_even = img[even];
            i_odd  = img[odd];

            r_tmp = r_even + r_odd;
            r_odd = r_even - r_odd;
            r_even = r_tmp;

            i_tmp = i_even + i_odd;
            i_odd = i_even - i_odd;
            i_even = i_tmp;

            real[even] = r_even;
            real[odd]  = r_odd;
            img[even]  = i_even;
            img[odd]   = i_odd;

            rootindex = (even << log) & bitmask;
            if (rootindex) {
                const double tw_r = real_twid[rootindex];
                const double tw_i = img_twid[rootindex];

                r_tmp = tw_r * r_odd - tw_i * i_odd;
                i_tmp = tw_r * i_odd + tw_i * r_odd;

                real[odd] = r_tmp;
                img[odd]  = i_tmp;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
