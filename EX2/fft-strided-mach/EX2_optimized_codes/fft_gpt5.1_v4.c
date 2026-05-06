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

    /* Iterative FFT, parallelized over odd indices within each span */
    for (span = FFT_SIZE >> 1; span; span >>= 1, log++) {
        int s = span;
        int l = log;

        /* Parallelize inner loop; iterations are independent for fixed span/log */
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int odd = s; odd < FFT_SIZE; odd++) {
            int o = odd | s;
            int e = o ^ s;

            double real_even = real[e];
            double real_odd  = real[o];
            double img_even  = img[e];
            double img_odd   = img[o];

            /* Butterfly on real part */
            double treal = real_even + real_odd;
            double ureal = real_even - real_odd;
            real[o] = ureal;
            real[e] = treal;

            /* Butterfly on imaginary part */
            double timg = img_even + img_odd;
            double uimg = img_even - img_odd;
            img[o] = uimg;
            img[e] = timg;

            /* Twiddle multiplication */
            int rootindex = (e << l) & (FFT_SIZE - 1);
            if (rootindex) {
                double wr = real_twid[rootindex];
                double wi = img_twid[rootindex];

                double xr = real[o];
                double xi = img[o];

                double tmp_real = wr * xr - wi * xi;
                double tmp_img  = wr * xi + wi * xr;

                real[o] = tmp_real;
                img[o]  = tmp_img;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
