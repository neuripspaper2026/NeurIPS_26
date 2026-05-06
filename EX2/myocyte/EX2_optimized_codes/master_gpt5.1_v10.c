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
    fp JCaDyad;
    fp JCaSL;
    fp JCaCyt;

    // offset pointers
    int initvalu_offset_ecc;   // 46 points
    int parameter_offset_ecc;
    int initvalu_offset_Dyad;  // 15 points
    int parameter_offset_Dyad;
    int initvalu_offset_SL;    // 15 points
    int parameter_offset_SL;
    int initvalu_offset_Cyt;   // 15 poitns
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

    // precompute scaled Ca values (from [mM] to [uM])
    const fp Ca_scale = (fp)1e3;
    CaDyad = initvalu[35] * Ca_scale;
    CaSL   = initvalu[36] * Ca_scale;
    CaCyt  = initvalu[37] * Ca_scale;

    // cam function offsets and parameter offsets
    initvalu_offset_Dyad = 46; // 15 points
    parameter_offset_Dyad = 1;
    initvalu_offset_SL = 61;   // 15 points
    parameter_offset_SL = 6;
    initvalu_offset_Cyt = 76;  // 15 poitns
    parameter_offset_Cyt = 11;

    // call cam for Dyad, SL, Cyt
    JCaDyad = cam(timeinst, initvalu, initvalu_offset_Dyad, parameter,
                  parameter_offset_Dyad, finavalu, CaDyad);

    JCaSL = cam(timeinst, initvalu, initvalu_offset_SL, parameter,
                parameter_offset_SL, finavalu, CaSL);

    JCaCyt = cam(timeinst, initvalu, initvalu_offset_Cyt, parameter,
                 parameter_offset_Cyt, finavalu, CaCyt);

    // final adjustments
    fin(initvalu, initvalu_offset_ecc, initvalu_offset_Dyad, initvalu_offset_SL,
        initvalu_offset_Cyt, parameter, finavalu, JCaDyad, JCaSL, JCaCyt);

    // make sure function does not return NANs and INFs
    // EQUATIONS is assumed to be reasonably large; use OpenMP to parallelize
    // the cleanup loop if available. The loop is embarrassingly parallel and
    // each iteration is independent.
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < EQUATIONS; i++) {
        const fp val = finavalu[i];
        if (isnan(val)) {
            finavalu[i] = (fp)0.0001; // for NAN set rate of change to 0.0001
        } else if (isinf(val)) {
            finavalu[i] = (fp)0.0001; // for INF set rate of change to 0.0001
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &master_end);
    master_kernel_time += (master_end.tv_sec - master_start.tv_sec) +
                          (master_end.tv_nsec - master_start.tv_nsec) / 1e9;
}
