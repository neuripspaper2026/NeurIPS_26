#include <time.h>
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]){
    int even, odd, span, log, rootindex;
    double temp_real, temp_img;
    log = 0;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for(span=FFT_SIZE>>1; span; span>>=1, log++){
        for(odd=span; odd<FFT_SIZE; odd++){
            odd |= span;
            even = odd ^ span;

            // Combine real and imaginary calculations to reduce redundant memory accesses
            temp_real = real[even] + real[odd];
            temp_img = img[even] + img[odd];
            
            real[odd] = real[even] - real[odd];
            img[odd] = img[even] - img[odd];
            
            real[even] = temp_real;
            img[even] = temp_img;

            rootindex = (even<<log) & (FFT_SIZE - 1);
            if(rootindex){
                temp_real = real_twid[rootindex] * real[odd] -
                    img_twid[rootindex]  * img[odd];
                temp_img = real_twid[rootindex]*img[odd] +
                    img_twid[rootindex]*real[odd];
                real[odd] = temp_real;
                img[odd] = temp_img;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
