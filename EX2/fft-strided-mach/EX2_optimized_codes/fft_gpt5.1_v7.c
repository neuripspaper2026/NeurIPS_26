#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]){
    int even, odd, span, log;
    double temp;
    int rootindex;
    log = 0;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    outer: for (span = FFT_SIZE >> 1; span; span >>= 1, log++) {
#ifdef _OPENMP
        /* Parallelize over all butterflies in this span.
         * Each (even,odd) pair is unique and written exactly once. */
#pragma omp parallel for private(odd, even, temp, rootindex) schedule(static)
#endif
        for (odd = 0; odd < FFT_SIZE; odd++) {
            /* Original loop iterated odd from span..FFT_SIZE and then did odd |= span.
             * Here we directly construct odd indices that have the span bit set. */
            if ((odd & span) == 0)
                continue;

            even = odd ^ span;

            /* Butterfly on real part */
            temp      = real[even] + real[odd];
            real[odd] = real[even] - real[odd];
            real[even]= temp;

            /* Butterfly on imaginary part */
            temp      = img[even] + img[odd];
            img[odd]  = img[even] - img[odd];
            img[even] = temp;

            /* Twiddle multiplication */
            rootindex = (even << log) & (FFT_SIZE - 1);
            if (rootindex) {
                double r_odd = real[odd];
                double i_odd = img[odd];
                double tw_r  = real_twid[rootindex];
                double tw_i  = img_twid[rootindex];

                temp      = tw_r * r_odd - tw_i * i_odd;
                img[odd]  = tw_r * i_odd + tw_i * r_odd;
                real[odd] = temp;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
