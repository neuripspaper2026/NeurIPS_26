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
    //	VARIABLES
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
    //	COMPUTATION
    //=====================================================================

    // decoded input parameters
    BtotDyad      = __ldg(parameter + 2); // parameter[2]
    CaMKIItotDyad = __ldg(parameter + 3); // parameter[3]

    // set variables (constants hoisted and using fp literals)
    Vmyo   = (fp)2.1454e-11;  // [L]
    Vdyad  = (fp)1.7790e-14;  // [L]
    VSL    = (fp)6.6013e-13;  // [L]
    kSLmyo = (fp)8.587e-15;   // [L/msec]
    k0Boff = (fp)0.0014;      // [s^-1]
    k0Bon  = k0Boff / (fp)0.2;  // [uM^-1 s^-1] kon = koff/Kd
    k2Boff = k0Boff / (fp)100.0; // [s^-1]
    k2Bon  = k0Bon;            // [uM^-1 s^-1]
    k4Bon  = k0Bon;            // [uM^-1 s^-1]

    const fp scale_JCa = (fp)1e-3;

    // ADJUST ECC incorporate Ca buffering from CaM, convert JCaCyt from uM/msec to mM/msec
    finavalu[initvalu_offset_ecc + 35] += scale_JCa * JCaDyad;
    finavalu[initvalu_offset_ecc + 36] += scale_JCa * JCaSL;
    finavalu[initvalu_offset_ecc + 37] += scale_JCa * JCaCyt;

    // Preload frequently used Dyad and SL values into registers (read-only)
    const int offDyad = initvalu_offset_Dyad;
    const int offSL   = initvalu_offset_SL;
    const int offCyt  = initvalu_offset_Cyt;

    const fp Dyad0  = __ldg(initvalu + offDyad + 0);
    const fp Dyad1  = __ldg(initvalu + offDyad + 1);
    const fp Dyad2  = __ldg(initvalu + offDyad + 2);
    const fp Dyad3  = __ldg(initvalu + offDyad + 3);
    const fp Dyad4  = __ldg(initvalu + offDyad + 4);
    const fp Dyad5  = __ldg(initvalu + offDyad + 5);
    const fp Dyad6  = __ldg(initvalu + offDyad + 6);
    const fp Dyad7  = __ldg(initvalu + offDyad + 7);
    const fp Dyad8  = __ldg(initvalu + offDyad + 8);
    const fp Dyad9  = __ldg(initvalu + offDyad + 9);
    const fp Dyad12 = __ldg(initvalu + offDyad + 12);
    const fp Dyad13 = __ldg(initvalu + offDyad + 13);
    const fp Dyad14 = __ldg(initvalu + offDyad + 14);

    const fp SL0 = __ldg(initvalu + offSL + 0);
    const fp SL1 = __ldg(initvalu + offSL + 1);
    const fp SL2 = __ldg(initvalu + offSL + 2);

    const fp Cyt0 = __ldg(initvalu + offCyt + 0);
    const fp Cyt1 = __ldg(initvalu + offCyt + 1);
    const fp Cyt2 = __ldg(initvalu + offCyt + 2);

    // incorporate CaM diffusion between compartments
    CaMtotDyad =
        Dyad0  + Dyad1  + Dyad2  + Dyad3  + Dyad4  + Dyad5 +
        CaMKIItotDyad * (Dyad6 + Dyad7 + Dyad8 + Dyad9) +
        Dyad12 + Dyad13 + Dyad14;

    Bdyad = BtotDyad - CaMtotDyad; // [uM dyad]

    // CaM fluxes between Dyad and SL (with fused scaling)
    const fp Bdyad_k0Bon = Bdyad * k0Bon;
    const fp Bdyad_k2Bon = Bdyad * k2Bon;
    const fp Bdyad_k4Bon = Bdyad * k4Bon;

    J_cam_dyadSL =
        scale_JCa *
        (k0Boff * Dyad0 - Bdyad_k0Bon * SL0); // [uM/msec dyad]

    J_ca2cam_dyadSL =
        scale_JCa *
        (k2Boff * Dyad1 - Bdyad_k2Bon * SL1); // [uM/msec dyad]

    J_ca4cam_dyadSL =
        scale_JCa *
        (k2Boff * Dyad2 - Bdyad_k4Bon * SL2); // [uM/msec dyad]

    // CaM fluxes between SL and myoplasm
    const fp SLmCyt0 = SL0 - Cyt0;
    const fp SLmCyt1 = SL1 - Cyt1;
    const fp SLmCyt2 = SL2 - Cyt2;

    J_cam_SLmyo    = kSLmyo * SLmCyt0; // [umol/msec]
    J_ca2cam_SLmyo = kSLmyo * SLmCyt1; // [umol/msec]
    J_ca4cam_SLmyo = kSLmyo * SLmCyt2; // [umol/msec]

    const fp Vdyad_over_VSL = Vdyad / VSL;
    const fp invVSL         = (fp)1.0 / VSL;
    const fp invVmyo        = (fp)1.0 / Vmyo;

    // ADJUST CAM Dyad
    finavalu[offDyad + 0] -= J_cam_dyadSL;
    finavalu[offDyad + 1] -= J_ca2cam_dyadSL;
    finavalu[offDyad + 2] -= J_ca4cam_dyadSL;

    // ADJUST CAM SL
    finavalu[offSL + 0] += J_cam_dyadSL * Vdyad_over_VSL - J_cam_SLmyo * invVSL;
    finavalu[offSL + 1] += J_ca2cam_dyadSL * Vdyad_over_VSL - J_ca2cam_SLmyo * invVSL;
    finavalu[offSL + 2] += J_ca4cam_dyadSL * Vdyad_over_VSL - J_ca4cam_SLmyo * invVSL;

    // ADJUST CAM Cyt
    finavalu[offCyt + 0] += J_cam_SLmyo * invVmyo;
    finavalu[offCyt + 1] += J_ca2cam_SLmyo * invVmyo;
    finavalu[offCyt + 2] += J_ca4cam_SLmyo * invVmyo;
}
