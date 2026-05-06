#include <stdio.h>
#include <stdlib.h>
#include "../../common_MachSuite/support.h"

#define FFT_SIZE 1024
#define twoPI 6.28318530717959

void fft(double real[FFT_SIZE], double img[FFT_SIZE], double real_twid[FFT_SIZE/2], double img_twid[FFT_SIZE/2]);
void reset_fft_strided_kernel_time(void);
double get_fft_strided_kernel_time(void);


////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
        double real[FFT_SIZE];
        double img[FFT_SIZE];
        double real_twid[FFT_SIZE/2];
        double img_twid[FFT_SIZE/2];
};
