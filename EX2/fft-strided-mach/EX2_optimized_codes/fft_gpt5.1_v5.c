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

    outer: for (span = FFT_SIZE >> 1; span; span >>= 1, log++) {
#ifdef _OPENMP
        /* Parallelize over the butterflies within this span.
         * Each (even, odd) pair is uniquely owned by one iteration of odd
         * and hence by one thread, avoiding data races.
         */
#pragma omp parallel
        {
#pragma omp for schedule(static)
#endif
        inner: for (int odd = span; odd < FFT_SIZE; odd++) {
                int idx = odd | span;
                int even = idx ^ span;

                /* Butterfly for real part */
                double real_even = real[even];
                double real_odd  = real[idx];
                double temp_real = real_even + real_odd;
                real[idx]  = real_even - real_odd;
                real[even] = temp_real;

                /* Butterfly for imaginary part */
                double img_even = img[even];
                double img_odd  = img[idx];
                double temp_img = img_even + img_odd;
                img[idx]  = img_even - img_odd;
                img[even] = temp_img;

                /* Twiddle multiplication */
                int rootindex = (even << log) & (FFT_SIZE - 1);
                if (rootindex) {
                    double r_twid = real_twid[rootindex];
                    double i_twid = img_twid[rootindex];

                    double r_odd = real[idx];
                    double i_odd = img[idx];

                    double tw_real = r_twid * r_odd - i_twid * i_odd;
                    double tw_img  = r_twid * i_odd + i_twid * r_odd;

                    real[idx] = tw_real;
                    img[idx]  = tw_img;
                }
            }
#ifdef _OPENMP
        }
#endif
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
