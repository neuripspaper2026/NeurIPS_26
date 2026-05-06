#include <math.h>
#include <time.h>

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
    const int initvalu_offset_ecc = 0;   // 46 points
    const int parameter_offset_ecc = 0;
    const int initvalu_offset_Dyad = 46; // 15 points
    const int parameter_offset_Dyad = 1;
    const int initvalu_offset_SL = 61;   // 15 points
    const int parameter_offset_SL = 6;
    const int initvalu_offset_Cyt = 76;  // 15 points
    const int parameter_offset_Cyt = 11;

    // module parameters (from ECC model, *** Converting from [mM] to [uM] ***)
    const fp CaDyad = initvalu[35] * (fp)1e3;
    const fp CaSL = initvalu[36] * (fp)1e3;
    const fp CaCyt = initvalu[37] * (fp)1e3;

    // ecc function
    ecc(timeinst, initvalu, initvalu_offset_ecc, parameter,
        parameter_offset_ecc, finavalu);

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
    for (int i = 0; i < EQUATIONS; ++i) {
        fp v = finavalu[i];
        if (isnan(v) || isinf(v)) {
            finavalu[i] = (fp)0.0001; // for NAN/INF set rate of change to 0.0001
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &master_end);
    master_kernel_time += (master_end.tv_sec - master_start.tv_sec) +
                          (master_end.tv_nsec - master_start.tv_nsec) / 1e9;
}
