#include <time.h>
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]){
    int even, odd, span, log, rootindex;
    double temp_real_odd, temp_img_odd;
    double real_even, img_even, real_odd, img_odd;
    double twid_real, twid_img;
    log = 0;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    outer:for(span=FFT_SIZE>>1; span; span>>=1, log++){
        inner:for(odd=span; odd<FFT_SIZE; odd++){
            odd |= span;
            even = odd ^ span;

            // Load values to local variables to reduce memory access
            real_even = real[even];
            img_even = img[even];
            real_odd = real[odd];
            img_odd = img[odd];

            // Perform butterfly operations
            real[odd] = real_even - real_odd;
            real[even] = real_even + real_odd;

            img[odd] = img_even - img_odd;
            img[even] = img_even + img_odd;

            rootindex = (even<<log) & (FFT_SIZE - 1);
            if(rootindex){
                // Load twiddle factors
                twid_real = real_twid[rootindex];
                twid_img = img_twid[rootindex];
                
                // Calculate twiddle factor multiplication with reduced memory access
                temp_real_odd = twid_real * real[odd] - twid_img * img[odd];
                temp_img_odd = twid_real * img[odd] + twid_img * real[odd];
                
                real[odd] = temp_real_odd;
                img[odd] = temp_img_odd;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
