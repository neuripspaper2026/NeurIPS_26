#include <time.h>
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]){
    int even, odd, span, log, rootindex;
    double temp_real, temp_img;
    double real_even, img_even, real_odd, img_odd;
    double real_twid_val, img_twid_val;
    log = 0;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    outer:for(span=FFT_SIZE>>1; span; span>>=1, log++){
        inner:for(odd=span; odd<FFT_SIZE; odd++){
            odd |= span;
            even = odd ^ span;

            real_even = real[even];
            img_even = img[even];
            real_odd = real[odd];
            img_odd = img[odd];

            temp_real = real_even + real_odd;
            real_odd = real_even - real_odd;
            real_even = temp_real;

            temp_img = img_even + img_odd;
            img_odd = img_even - img_odd;
            img_even = temp_img;

            rootindex = (even<<log) & (FFT_SIZE - 1);
            if(rootindex){
                real_twid_val = real_twid[rootindex];
                img_twid_val = img_twid[rootindex];
                
                temp_real = real_twid_val * real_odd - img_twid_val * img_odd;
                img_odd = real_twid_val * img_odd + img_twid_val * real_odd;
                real_odd = temp_real;
            }

            real[even] = real_even;
            img[even] = img_even;
            real[odd] = real_odd;
            img[odd] = img_odd;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
