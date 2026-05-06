#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "../define.h"

void ecc(fp timeinst, fp *initvalu, int initvalu_offset, fp *parameter,
         int parameter_offset, fp *finavalu);
fp cam(fp timeinst, fp *initvalu, int initvalu_offset, fp *parameter,
       int parameter_offset, fp *finavalu, fp Ca);
void fin(fp *initvalu, int initvalu_offset_ecc, int initvalu_offset_Dyad,
         int initvalu_offset_SL, int initvalu_offset_Cyt, fp *parameter,
         fp *finavalu, fp JCaDyad, fp JCaSL, fp JCaCyt);

double master_kernel_time = 0.0;

void master(fp timeinst, fp *initvalu, fp *parameter, fp *finavalu, int mode) {
    struct timespec master_start, master_end;
    clock_gettime(CLOCK_MONOTONIC, &master_start);

    fp JCaDyad, JCaSL, JCaCyt;
    fp CaDyad, CaSL, CaCyt;

    ecc(timeinst, initvalu, 0, parameter, 0, finavalu);

    CaDyad = initvalu[35] * 1e3;
    CaSL = initvalu[36] * 1e3;
    CaCyt = initvalu[37] * 1e3;

#ifdef _OPENMP
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            JCaDyad = cam(timeinst, initvalu, 46, parameter, 1, finavalu, CaDyad);
        }
        #pragma omp section
        {
            JCaSL = cam(timeinst, initvalu, 61, parameter, 6, finavalu, CaSL);
        }
        #pragma omp section
        {
            JCaCyt = cam(timeinst, initvalu, 76, parameter, 11, finavalu, CaCyt);
        }
    }
#else
    JCaDyad = cam(timeinst, initvalu, 46, parameter, 1, finavalu, CaDyad);
    JCaSL = cam(timeinst, initvalu, 61, parameter, 6, finavalu, CaSL);
    JCaCyt = cam(timeinst, initvalu, 76, parameter, 11, finavalu, CaCyt);
#endif

    fin(initvalu, 0, 46, 61, 76, parameter, finavalu, JCaDyad, JCaSL, JCaCyt);

#ifdef _OPENMP
    #pragma omp parallel for simd
#endif
    for (int i = 0; i < EQUATIONS; i++) {
        if (isnan(finavalu[i]) == 1) {
            finavalu[i] = 0.0001;
        } else if (isinf(finavalu[i]) == 1) {
            finavalu[i] = 0.0001;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &master_end);
    master_kernel_time += (master_end.tv_sec - master_start.tv_sec) +
                          (master_end.tv_nsec - master_start.tv_nsec) / 1e9;
}
