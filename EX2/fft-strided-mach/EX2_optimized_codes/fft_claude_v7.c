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
    for(span=FFT_SIZE>>1; span; span>>=1, log++){
        const int log_const = log;
        const int span_const = span;
        
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int base=span_const; base<FFT_SIZE; base+=(span_const<<1)){
            int odd = base;
            int even = base ^ span_const;
            
            double temp_real = real[even] + real[odd];
            double temp_odd_real = real[even] - real[odd];
            
            double temp_img = img[even] + img[odd];
            double temp_odd_img = img[even] - img[odd];
            
            real[even] = temp_real;
            img[even] = temp_img;
            
            int rootindex = (even<<log_const) & (FFT_SIZE - 1);
            if(rootindex){
                double rt = real_twid[rootindex];
                double it = img_twid[rootindex];
                real[odd] = rt * temp_odd_real - it * temp_odd_img;
                img[odd] = rt * temp_odd_img + it * temp_odd_real;
            } else {
                real[odd] = temp_odd_real;
                img[odd] = temp_odd_img;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
