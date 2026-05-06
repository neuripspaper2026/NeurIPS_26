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
        int span_local = span;
        int log_local  = log;

        /* Each iteration of 'odd' touches only indices 'odd' and 'even=odd^span',
           which form disjoint pairs when 'odd' runs over this range, so we can
           safely parallelize the loop. */
        #pragma omp parallel for schedule(static)
        for (int odd = span_local; odd < FFT_SIZE; odd++) {
            int odd_idx  = odd | span_local;
            int even_idx = odd_idx ^ span_local;

            double r_even = real[even_idx];
            double r_odd  = real[odd_idx];

            double i_even = img[even_idx];
            double i_odd  = img[odd_idx];

            double temp_r = r_even + r_odd;
            double new_r_odd = r_even - r_odd;

            double temp_i = i_even + i_odd;
            double new_i_odd = i_even - i_odd;

            real[even_idx] = temp_r;
            img[even_idx]  = temp_i;

            int rootindex = (even_idx << log_local) & (FFT_SIZE - 1);
            if (rootindex) {
                double twr = real_twid[rootindex];
                double twi = img_twid[rootindex];

                double t_r = twr * new_r_odd - twi * new_i_odd;
                double t_i = twr * new_i_odd + twi * new_r_odd;

                real[odd_idx] = t_r;
                img[odd_idx]  = t_i;
            } else {
                real[odd_idx] = new_r_odd;
                img[odd_idx]  = new_i_odd;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
