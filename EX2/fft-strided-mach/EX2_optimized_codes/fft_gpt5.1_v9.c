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
        int odd;

        /* Parallelize the butterfly operations within each stage.
         * Each (even,odd) pair is independent for a given span/log.
         */
        #ifdef _OPENMP
        #pragma omp parallel for private(odd) schedule(static)
        #endif
        for (odd = span; odd < FFT_SIZE; odd++) {
            int local_odd = odd | span;
            int even = local_odd ^ span;
            int rootindex;
            double temp;
            double real_even = real[even];
            double real_odd  = real[local_odd];
            double img_even  = img[even];
            double img_odd   = img[local_odd];

            /* Real part butterfly */
            temp       = real_even + real_odd;
            real_odd   = real_even - real_odd;
            real_even  = temp;

            /* Imaginary part butterfly */
            temp       = img_even + img_odd;
            img_odd    = img_even - img_odd;
            img_even   = temp;

            /* Store updated even values back immediately */
            real[even] = real_even;
            img[even]  = img_even;

            /* Twiddle multiplication for odd index */
            rootindex = (even << log) & (FFT_SIZE - 1);
            if (rootindex) {
                double rt = real_twid[rootindex];
                double it = img_twid[rootindex];
                double ro = real_odd;
                double io = img_odd;

                temp   = rt * ro - it * io;
                img_odd = rt * io + it * ro;
                real_odd = temp;
            }

            /* Store updated odd values back */
            real[local_odd] = real_odd;
            img[local_odd]  = img_odd;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
