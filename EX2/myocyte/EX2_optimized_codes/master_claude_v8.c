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
    int initvalu_offset_ecc = 0;
    int parameter_offset_ecc = 0;
    int initvalu_offset_Dyad = 46;
    int parameter_offset_Dyad = 1;
    int initvalu_offset_SL = 61;
    int parameter_offset_SL = 6;
    int initvalu_offset_Cyt = 76;
    int parameter_offset_Cyt = 11;

    // module parameters
    fp CaDyad;
    fp CaSL;
    fp CaCyt;

    // ecc function
    ecc(timeinst, initvalu, initvalu_offset_ecc, parameter,
        parameter_offset_ecc, finavalu);

    // Precompute Ca values once
    CaDyad = initvalu[35] * 1e3;
    CaSL = initvalu[36] * 1e3;
    CaCyt = initvalu[37] * 1e3;

    // cam function for Dyad
    JCaDyad = cam(timeinst, initvalu, initvalu_offset_Dyad, parameter,
                  parameter_offset_Dyad, finavalu, CaDyad);

    // cam function for SL
    JCaSL = cam(timeinst, initvalu, initvalu_offset_SL, parameter,
                parameter_offset_SL, finavalu, CaSL);

    // cam function for Cyt
    JCaCyt = cam(timeinst, initvalu, initvalu_offset_Cyt, parameter,
                 parameter_offset_Cyt, finavalu, CaCyt);

    // final adjustments
    fin(initvalu, initvalu_offset_ecc, initvalu_offset_Dyad, initvalu_offset_SL,
        initvalu_offset_Cyt, parameter, finavalu, JCaDyad, JCaSL, JCaCyt);

    // make sure function does not return NANs and INFs
    #pragma omp parallel for if(EQUATIONS > 100) schedule(static)
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
