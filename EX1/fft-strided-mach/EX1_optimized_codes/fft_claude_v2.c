#include <time.h>
#include "../fft.h"

static double fft_strided_kernel_time_acc = 0.0;

void reset_fft_strided_kernel_time(void) { fft_strided_kernel_time_acc = 0.0; }
double get_fft_strided_kernel_time(void) { return fft_strided_kernel_time_acc; }

void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]){
    int even, odd, span, log, rootindex;
    double temp;
    log = 0;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    outer:for(span=FFT_SIZE>>1; span; span>>=1, log++){
        inner:for(odd=span; odd<FFT_SIZE; odd++){
            odd |= span;
            even = odd ^ span;

            double real_even = real[even];
            double real_odd = real[odd];
            double img_even = img[even];
            double img_odd = img[odd];

            temp = real_even + real_odd;
            real_odd = real_even - real_odd;
            real_even = temp;

            temp = img_even + img_odd;
            img_odd = img_even - img_odd;
            img_even = temp;

            rootindex = (even<<log) & (FFT_SIZE - 1);
            if(rootindex){
                double real_tw = real_twid[rootindex];
                double img_tw = img_twid[rootindex];
                temp = real_tw * real_odd - img_tw * img_odd;
                img_odd = real_tw * img_odd + img_tw * real_odd;
                real_odd = temp;
            }

            real[even] = real_even;
            real[odd] = real_odd;
            img[even] = img_even;
            img[odd] = img_odd;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
