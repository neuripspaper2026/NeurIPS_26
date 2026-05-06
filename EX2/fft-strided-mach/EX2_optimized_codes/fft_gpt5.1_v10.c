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

    /* Outer loop over spans; each span level can be parallelized */
    for (span = FFT_SIZE >> 1; span; span >>= 1, log++) {

#ifdef _OPENMP
        #pragma omp parallel
        {
            int odd;
            int even;
            int rootindex;
            double temp;
            /* Distribute odd-index work across threads for this span */
            #pragma omp for schedule(static)
            for (odd = span; odd < FFT_SIZE; odd++) {
                int o = odd | span;
                even = o ^ span;

                /* Butterfly on real part */
                double real_even = real[even];
                double real_odd  = real[o];
                double real_sum  = real_even + real_odd;
                double real_diff = real_even - real_odd;
                real[even] = real_sum;
                real[o]    = real_diff;

                /* Butterfly on imaginary part */
                double img_even = img[even];
                double img_odd  = img[o];
                double img_sum  = img_even + img_odd;
                double img_diff = img_even - img_odd;
                img[even] = img_sum;
                img[o]    = img_diff;

                rootindex = (even << log) & (FFT_SIZE - 1);

                if (rootindex) {
                    /* Twiddle multiplication with temporary locals to help vectorization */
                    double r_tw = real_twid[rootindex];
                    double i_tw = img_twid[rootindex];
                    double r_o  = real[o];
                    double i_o  = img[o];

                    temp   = r_tw * r_o - i_tw * i_o;
                    img[o] = r_tw * i_o + i_tw * r_o;
                    real[o]= temp;
                }
            }
        }
#else
        {
            int odd;
            int even;
            int rootindex;
            double temp;
            for (odd = span; odd < FFT_SIZE; odd++) {
                int o = odd | span;
                even = o ^ span;

                /* Butterfly on real part */
                double real_even = real[even];
                double real_odd  = real[o];
                double real_sum  = real_even + real_odd;
                double real_diff = real_even - real_odd;
                real[even] = real_sum;
                real[o]    = real_diff;

                /* Butterfly on imaginary part */
                double img_even = img[even];
                double img_odd  = img[o];
                double img_sum  = img_even + img_odd;
                double img_diff = img_even - img_odd;
                img[even] = img_sum;
                img[o]    = img_diff;

                rootindex = (even << log) & (FFT_SIZE - 1);

                if (rootindex) {
                    double r_tw = real_twid[rootindex];
                    double i_tw = img_twid[rootindex];
                    double r_o  = real[o];
                    double i_o  = img[o];

                    temp   = r_tw * r_o - i_tw * i_o;
                    img[o] = r_tw * i_o + i_tw * r_o;
                    real[o]= temp;
                }
            }
        }
#endif
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
