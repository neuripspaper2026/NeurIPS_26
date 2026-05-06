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
        #pragma omp parallel if(span_const >= 64)
#endif
        {
            int odd, even, rootindex;
            double temp, temp_real_odd, temp_img_odd;
            double twid_real, twid_img;
            
#ifdef _OPENMP
            #pragma omp for schedule(static) nowait
#endif
            for(odd=span_const; odd<FFT_SIZE; odd++){
                int odd_val = odd | span_const;
                even = odd_val ^ span_const;

                double real_even = real[even];
                double real_odd = real[odd_val];
                double img_even = img[even];
                double img_odd = img[odd_val];

                temp = real_even + real_odd;
                temp_real_odd = real_even - real_odd;
                real[even] = temp;

                temp = img_even + img_odd;
                temp_img_odd = img_even - img_odd;
                img[even] = temp;

                rootindex = (even << log_const) & (FFT_SIZE - 1);
                if(rootindex){
                    twid_real = real_twid[rootindex];
                    twid_img = img_twid[rootindex];
                    
                    temp = twid_real * temp_real_odd - twid_img * temp_img_odd;
                    temp_img_odd = twid_real * temp_img_odd + twid_img * temp_real_odd;
                    temp_real_odd = temp;
                }
                
                real[odd_val] = temp_real_odd;
                img[odd_val] = temp_img_odd;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
