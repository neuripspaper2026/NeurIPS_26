__device__ __forceinline__ void kernel_fin(fp * __restrict__ initvalu,
                                           int initvalu_offset_ecc,
                                           int initvalu_offset_Dyad,
                                           int initvalu_offset_SL,
                                           int initvalu_offset_Cyt,
                                           fp * __restrict__ parameter,
                                           fp * __restrict__ finavalu,
                                           fp JCaDyad,
                                           fp JCaSL,
                                           fp JCaCyt) {

    //=====================================================================
    //  VARIABLES
    //=====================================================================

    // decoded input parameters
    fp BtotDyad;      //
    fp CaMKIItotDyad; //

    // compute variables
    fp Vmyo;  // [L]
    fp Vdyad; // [L]
    fp VSL;   // [L]
    fp kSLmyo; // [L/msec]
    fp k0Boff; // [s^-1]
    fp k0Bon;  // [uM^-1 s^-1] kon = koff/Kd
    fp k2Boff; // [s^-1]
    fp k2Bon;  // [uM^-1 s^-1]
    fp k4Bon;  // [uM^-1 s^-1]
    fp CaMtotDyad;
    fp Bdyad;           // [uM dyad]
    fp J_cam_dyadSL;    // [uM/msec dyad]
    fp J_ca2cam_dyadSL; // [uM/msec dyad]
    fp J_ca4cam_dyadSL; // [uM/msec dyad]
    fp J_cam_SLmyo;     // [umol/msec]
    fp J_ca2cam_SLmyo;  // [umol/msec]
    fp J_ca4cam_SLmyo;  // [umol/msec]

    //=====================================================================
    //  COMPUTATION
    //=====================================================================

    // decoded input parameters (use read-only cache)
    BtotDyad      = __ldg(&parameter[2]);
    CaMKIItotDyad = __ldg(&parameter[3]);

    // set variables (hoisted constants)
    Vmyo   = (fp)2.1454e-11;  // [L]
    Vdyad  = (fp)1.7790e-14;  // [L]
    VSL    = (fp)6.6013e-13;  // [L]
    kSLmyo = (fp)8.587e-15;   // [L/msec]
    k0Boff = (fp)0.0014;      // [s^-1]
    k0Bon  = k0Boff / (fp)0.2;// [uM^-1 s^-1] kon = koff/Kd
    k2Boff = k0Boff / (fp)100.0; // [s^-1]
    k2Bon  = k0Bon;           // [uM^-1 s^-1]
    k4Bon  = k0Bon;           // [uM^-1 s^-1]

    const fp SCALE_MM = (fp)1e-3;

    // load ECC Ca states (coalesced-friendly, cached)
    const int ecc35 = initvalu_offset_ecc + 35;
    const int ecc36 = initvalu_offset_ecc + 36;
    const int ecc37 = initvalu_offset_ecc + 37;

    const fp fin35 = __ldg(&finavalu[ecc35]);
    const fp fin36 = __ldg(&finavalu[ecc36]);
    const fp fin37 = __ldg(&finavalu[ecc37]);

    // ADJUST ECC incorporate Ca buffering from CaM, convert JCa* from uM/msec to mM/msec
    finavalu[ecc35] = fin35 + SCALE_MM * JCaDyad;
    finavalu[ecc36] = fin36 + SCALE_MM * JCaSL;
    finavalu[ecc37] = fin37 + SCALE_MM * JCaCyt;

    // precompute offsets for compartments
    const int offDyad = initvalu_offset_Dyad;
    const int offSL   = initvalu_offset_SL;
    const int offCyt  = initvalu_offset_Cyt;

    // load Dyad states used in CaMtotDyad (cached reads)
    const fp Dyad_0  = __ldg(&initvalu[offDyad + 0]);
    const fp Dyad_1  = __ldg(&initvalu[offDyad + 1]);
    const fp Dyad_2  = __ldg(&initvalu[offDyad + 2]);
    const fp Dyad_3  = __ldg(&initvalu[offDyad + 3]);
    const fp Dyad_4  = __ldg(&initvalu[offDyad + 4]);
    const fp Dyad_5  = __ldg(&initvalu[offDyad + 5]);
    const fp Dyad_6  = __ldg(&initvalu[offDyad + 6]);
    const fp Dyad_7  = __ldg(&initvalu[offDyad + 7]);
    const fp Dyad_8  = __ldg(&initvalu[offDyad + 8]);
    const fp Dyad_9  = __ldg(&initvalu[offDyad + 9]);
    const fp Dyad_12 = __ldg(&initvalu[offDyad + 12]);
    const fp Dyad_13 = __ldg(&initvalu[offDyad + 13]);
    const fp Dyad_14 = __ldg(&initvalu[offDyad + 14]);

    // incorporate CaM diffusion between compartments (fully unrolled)
    CaMtotDyad =
        Dyad_0 + Dyad_1 + Dyad_2 + Dyad_3 + Dyad_4 + Dyad_5 +
        CaMKIItotDyad * (Dyad_6 + Dyad_7 + Dyad_8 + Dyad_9) +
        Dyad_12 + Dyad_13 + Dyad_14;

    Bdyad = BtotDyad - CaMtotDyad; // [uM dyad]

    // load SL states needed (cached)
    const fp SL_0 = __ldg(&initvalu[offSL + 0]);
    const fp SL_1 = __ldg(&initvalu[offSL + 1]);
    const fp SL_2 = __ldg(&initvalu[offSL + 2]);

    // Dyad CaM species for fluxes (reuse already loaded where possible)
    const fp Dyad_0_loc = Dyad_0;
    const fp Dyad_1_loc = Dyad_1;
    const fp Dyad_2_loc = Dyad_2;

    // J_cam_* dyad<->SL fluxes (use FMA-friendly expressions)
    J_cam_dyadSL =
        SCALE_MM * (k0Boff * Dyad_0_loc - k0Bon * Bdyad * SL_0); // [uM/msec dyad]
    J_ca2cam_dyadSL =
        SCALE_MM * (k2Boff * Dyad_1_loc - k2Bon * Bdyad * SL_1); // [uM/msec dyad]
    J_ca4cam_dyadSL =
        SCALE_MM * (k2Boff * Dyad_2_loc - k4Bon * Bdyad * SL_2); // [uM/msec dyad]

    // SL <-> myo fluxes
    const fp Cyt_0 = __ldg(&initvalu[offCyt + 0]);
    const fp Cyt_1 = __ldg(&initvalu[offCyt + 1]);
    const fp Cyt_2 = __ldg(&initvalu[offCyt + 2]);

    const fp SL_minus_Cyt_0 = SL_0 - Cyt_0;
    const fp SL_minus_Cyt_1 = SL_1 - Cyt_1;
    const fp SL_minus_Cyt_2 = SL_2 - Cyt_2;

    J_cam_SLmyo    = kSLmyo * SL_minus_Cyt_0; // [umol/msec]
    J_ca2cam_SLmyo = kSLmyo * SL_minus_Cyt_1; // [umol/msec]
    J_ca4cam_SLmyo = kSLmyo * SL_minus_Cyt_2; // [umol/msec]

    //=====================================================================
    //  ADJUST STATES (use temporaries for better ILP)
    //=====================================================================

    // Dyad updates
    const int fDyad0 = offDyad + 0;
    const int fDyad1 = offDyad + 1;
    const int fDyad2 = offDyad + 2;

    const fp fDyad0_old = __ldg(&finavalu[fDyad0]);
    const fp fDyad1_old = __ldg(&finavalu[fDyad1]);
    const fp fDyad2_old = __ldg(&finavalu[fDyad2]);

    const fp fDyad0_new = fDyad0_old - J_cam_dyadSL;
    const fp fDyad1_new = fDyad1_old - J_ca2cam_dyadSL;
    const fp fDyad2_new = fDyad2_old - J_ca4cam_dyadSL;

    finavalu[fDyad0] = fDyad0_new;
    finavalu[fDyad1] = fDyad1_new;
    finavalu[fDyad2] = fDyad2_new;

    // SL updates (reuse volume ratios)
    const fp Vdyad_over_VSL = Vdyad / VSL;
    const fp inv_VSL        = (fp)1.0 / VSL;
    const fp inv_Vmyo       = (fp)1.0 / Vmyo;

    const int fSL0 = offSL + 0;
    const int fSL1 = offSL + 1;
    const int fSL2 = offSL + 2;

    const fp fSL0_old = __ldg(&finavalu[fSL0]);
    const fp fSL1_old = __ldg(&finavalu[fSL1]);
    const fp fSL2_old = __ldg(&finavalu[fSL2]);

    const fp SL_term0 = J_cam_dyadSL    * Vdyad_over_VSL - J_cam_SLmyo    * inv_VSL;
    const fp SL_term1 = J_ca2cam_dyadSL * Vdyad_over_VSL - J_ca2cam_SLmyo * inv_VSL;
    const fp SL_term2 = J_ca4cam_dyadSL * Vdyad_over_VSL - J_ca4cam_SLmyo * inv_VSL;

    finavalu[fSL0] = fSL0_old + SL_term0;
    finavalu[fSL1] = fSL1_old + SL_term1;
    finavalu[fSL2] = fSL2_old + SL_term2;

    // Cyt updates
    const int fCyt0 = offCyt + 0;
    const int fCyt1 = offCyt + 1;
    const int fCyt2 = offCyt + 2;

    const fp fCyt0_old = __ldg(&finavalu[fCyt0]);
    const fp fCyt1_old = __ldg(&finavalu[fCyt1]);
    const fp fCyt2_old = __ldg(&finavalu[fCyt2]);

    finavalu[fCyt0] = fCyt0_old + J_cam_SLmyo    * inv_Vmyo;
    finavalu[fCyt1] = fCyt1_old + J_ca2cam_SLmyo * inv_Vmyo;
    finavalu[fCyt2] = fCyt2_old + J_ca4cam_SLmyo * inv_Vmyo;
}
