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

    log = 0;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (span = FFT_SIZE >> 1; span; span >>= 1, log++) {
        const int span_local = span;
        const int log_local  = log;

        /* Parallelize the butterfly operations when OpenMP is available.
         * Each iteration of this loop works on distinct indices (even, odd),
         * so it is safe to parallelize. */
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int idx = 0; idx < FFT_SIZE; idx++) {
            int odd = idx | span_local;

            /* Only process valid odd indices within range. This replaces
             * the original "for (odd = span; odd < FFT_SIZE; odd++) { odd |= span; }"
             * pattern with a branch. */
            if (odd < span_local || odd >= FFT_SIZE)
                continue;

            int even = odd ^ span_local;

            double real_even = real[even];
            double real_odd  = real[odd];
            double img_even  = img[even];
            double img_odd   = img[odd];

            /* Butterfly for real part */
            double tmp_real_sum  = real_even + real_odd;
            double tmp_real_diff = real_even - real_odd;

            /* Butterfly for imag part */
            double tmp_img_sum   = img_even + img_odd;
            double tmp_img_diff  = img_even - img_odd;

            real[even] = tmp_real_sum;
            real[odd]  = tmp_real_diff;
            img[even]  = tmp_img_sum;
            img[odd]   = tmp_img_diff;

            int rootindex = (even << log_local) & (FFT_SIZE - 1);

            if (rootindex) {
                double rt = real_twid[rootindex];
                double it = img_twid[rootindex];

                double r_odd = real[odd];
                double i_odd = img[odd];

                double tw_real = rt * r_odd - it * i_odd;
                double tw_img  = rt * i_odd + it * r_odd;

                real[odd] = tw_real;
                img[odd]  = tw_img;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
