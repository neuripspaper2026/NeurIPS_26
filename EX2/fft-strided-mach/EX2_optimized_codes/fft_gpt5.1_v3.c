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
    for (span = FFT_SIZE >> 1; span; span >>= 1, log++) {
        int odd_start = span;
        int odd_end   = FFT_SIZE;

        /* Each (even,odd) pair is independent for a fixed span/log, so parallelize over odd. */
        #ifdef _OPENMP
        #pragma omp parallel for default(none) shared(real,img,real_twid,img_twid,span,log,odd_start,odd_end) schedule(static)
        #endif
        for (int odd = odd_start; odd < odd_end; odd++) {
            int o = odd | span;
            int even = o ^ span;

            double real_even = real[even];
            double real_odd  = real[o];
            double img_even  = img[even];
            double img_odd   = img[o];

            double temp_real = real_even + real_odd;
            double new_real_odd = real_even - real_odd;
            real[even] = temp_real;

            double temp_img = img_even + img_odd;
            double new_img_odd = img_even - img_odd;
            img[even] = temp_img;

            real[o] = new_real_odd;
            img[o]  = new_img_odd;

            int rootindex = (even << log) & (FFT_SIZE - 1);
            if (rootindex) {
                double twr = real_twid[rootindex];
                double twi = img_twid[rootindex];

                double r_odd = real[o];
                double i_odd = img[o];

                double r_tmp = twr * r_odd - twi * i_odd;
                double i_tmp = twr * i_odd + twi * r_odd;

                real[o] = r_tmp;
                img[o]  = i_tmp;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
