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
        int log_local = log;
        int span_local = span;
        
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for(int base=span_local; base<FFT_SIZE; base+=(span_local<<1)){
            for(int offset=0; offset<span_local; offset++){
                int odd = base + offset;
                int even = odd ^ span_local;

                double temp_real = real[even] + real[odd];
                double temp_img = img[even] + img[odd];
                double real_odd_new = real[even] - real[odd];
                double img_odd_new = img[even] - img[odd];

                int rootindex = (even<<log_local) & (FFT_SIZE - 1);
                
                if(rootindex){
                    double tw_real = real_twid[rootindex];
                    double tw_img = img_twid[rootindex];
                    double temp = tw_real * real_odd_new - tw_img * img_odd_new;
                    img_odd_new = tw_real * img_odd_new + tw_img * real_odd_new;
                    real_odd_new = temp;
                }

                real[even] = temp_real;
                img[even] = temp_img;
                real[odd] = real_odd_new;
                img[odd] = img_odd_new;
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    fft_strided_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
