#include <time.h>
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]){
    int even, odd, span, log, rootindex;
    double temp_real_odd, temp_img_odd;
    log = 0;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    outer:for(span=FFT_SIZE>>1; span; span>>=1, log++){
        inner:for(odd=span; odd<FFT_SIZE; odd++){
            odd |= span;
            even = odd ^ span;

            // Reduce memory accesses by storing computed values
            temp_real_odd = real[even] - real[odd];
            temp_img_odd = img[even] - img[odd];
            
            real[odd] = temp_real_odd;
            img[odd] = temp_img_odd;
            
            real[even] += real[odd];
            img[even] += img[odd];

            rootindex = (even<<log) & (FFT_SIZE - 1);
            if(rootindex){
                temp_real_odd = real_twid[rootindex] * real[odd] -
                    img_twid[rootindex]  * img[odd];
                temp_img_odd = real_twid[rootindex]*img[odd] +
                    img_twid[rootindex]*real[odd];
                real[odd] = temp_real_odd;
                img[odd] = temp_img_odd;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
