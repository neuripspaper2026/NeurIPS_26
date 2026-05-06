#include <time.h>
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE],
         double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]) {
    int even, odd, span, log, rootindex;
    double temp_real, temp_img;
    log = 0;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    outer:
    for (span = FFT_SIZE >> 1; span; span >>= 1, log++) {
        inner:
        for (odd = span; odd < FFT_SIZE; ) {
            odd |= span;
            even = odd ^ span;

            /* Butterfly on real parts */
            temp_real = real[even] + real[odd];
            real[odd] = real[even] - real[odd];
            real[even] = temp_real;

            /* Butterfly on imaginary parts */
            temp_img = img[even] + img[odd];
            img[odd] = img[even] - img[odd];
            img[even] = temp_img;

            /* Twiddle factor multiplication */
            rootindex = (even << log) & (FFT_SIZE - 1);
            if (rootindex) {
                const double tw_re = real_twid[rootindex];
                const double tw_im = img_twid[rootindex];
                const double r_odd = real[odd];
                const double i_odd = img[odd];

                temp_real = tw_re * r_odd - tw_im * i_odd;
                img[odd]  = tw_re * i_odd + tw_im * r_odd;
                real[odd] = temp_real;
            }

            /* Increment to next base index in this span group */
            odd++;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
