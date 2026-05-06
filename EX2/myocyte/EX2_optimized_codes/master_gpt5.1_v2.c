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

    // intermediate output on host
    fp JCaDyad = 0;
    fp JCaSL = 0;
    fp JCaCyt = 0;

    // offset pointers
    int initvalu_offset_ecc;   // 46 points
    int parameter_offset_ecc;
    int initvalu_offset_Dyad; // 15 points
    int parameter_offset_Dyad;
    int initvalu_offset_SL; // 15 points
    int parameter_offset_SL;
    int initvalu_offset_Cyt; // 15 poitns
    int parameter_offset_Cyt;

    // module parameters
    fp CaDyad; // from ECC model, *** Converting from [mM] to [uM] ***
    fp CaSL;   // from ECC model, *** Converting from [mM] to [uM] ***
    fp CaCyt;  // from ECC model, *** Converting from [mM] to [uM] ***

    // ecc function
    initvalu_offset_ecc = 0; // 46 points
    parameter_offset_ecc = 0;
    ecc(timeinst, initvalu, initvalu_offset_ecc, parameter,
        parameter_offset_ecc, finavalu);

    // Preload Ca values once (avoid repeated loads)
    CaDyad = initvalu[35] * (fp)1e3;
    CaSL   = initvalu[36] * (fp)1e3;
    CaCyt  = initvalu[37] * (fp)1e3;

    // cam function for Dyad, SL, Cyt
    initvalu_offset_Dyad = 46; // 15 points
    parameter_offset_Dyad = 1;
    initvalu_offset_SL = 61; // 15 points
    parameter_offset_SL = 6;
    initvalu_offset_Cyt = 76; // 15 poitns
    parameter_offset_Cyt = 11;

    // Three cam calls are independent; parallelize them
    #ifdef _OPENMP
    #pragma omp parallel sections default(none) shared(timeinst, initvalu, parameter, finavalu, \
                                                     CaDyad, CaSL, CaCyt,                      \
                                                     initvalu_offset_Dyad, parameter_offset_Dyad, \
                                                     initvalu_offset_SL, parameter_offset_SL,     \
                                                     initvalu_offset_Cyt, parameter_offset_Cyt,   \
                                                     JCaDyad, JCaSL, JCaCyt)
    {
        #pragma omp section
        {
            JCaDyad = cam(timeinst, initvalu, initvalu_offset_Dyad, parameter,
                         parameter_offset_Dyad, finavalu, CaDyad);
        }
        #pragma omp section
        {
            JCaSL = cam(timeinst, initvalu, initvalu_offset_SL, parameter,
                       parameter_offset_SL, finavalu, CaSL);
        }
        #pragma omp section
        {
            JCaCyt = cam(timeinst, initvalu, initvalu_offset_Cyt, parameter,
                        parameter_offset_Cyt, finavalu, CaCyt);
        }
    }
    #else
    JCaDyad = cam(timeinst, initvalu, initvalu_offset_Dyad, parameter,
                  parameter_offset_Dyad, finavalu, CaDyad);
    JCaSL = cam(timeinst, initvalu, initvalu_offset_SL, parameter,
                parameter_offset_SL, finavalu, CaSL);
    JCaCyt = cam(timeinst, initvalu, initvalu_offset_Cyt, parameter,
                 parameter_offset_Cyt, finavalu, CaCyt);
    #endif

    // final adjustments
    fin(initvalu, initvalu_offset_ecc, initvalu_offset_Dyad, initvalu_offset_SL,
        initvalu_offset_Cyt, parameter, finavalu, JCaDyad, JCaSL, JCaCyt);

    // make sure function does not return NANs and INFs
    // this loop is small; leave serial to avoid parallel overhead/contention
    for (int i = 0; i < EQUATIONS; i++) {
        fp v = finavalu[i];
        if (isnan(v) || isinf(v)) {
            finavalu[i] = (fp)0.0001; // set rate of change to 0.0001
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &master_end);
    master_kernel_time += (master_end.tv_sec - master_start.tv_sec) +
                          (master_end.tv_nsec - master_start.tv_nsec) / 1e9;
}
