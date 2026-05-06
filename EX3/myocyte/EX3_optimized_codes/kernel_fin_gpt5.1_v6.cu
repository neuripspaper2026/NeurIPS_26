__device__ void kernel_fin(fp * __restrict__ initvalu,
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
    const fp BtotDyad      = parameter[2];
    const fp CaMKIItotDyad = parameter[3];

    // precomputed constants
    const fp Vmyo   = (fp)2.1454e-11;  // [L]
    const fp Vdyad  = (fp)1.7790e-14;  // [L]
    const fp VSL    = (fp)6.6013e-13;  // [L]
    const fp invVmyo = (fp)1.0 / Vmyo;
    const fp invVSL  = (fp)1.0 / VSL;
    const fp Vdyad_over_VSL = Vdyad * invVSL;

    const fp k0Boff = (fp)0.0014;             // [s^-1]
    const fp k0Bon  = k0Boff / (fp)0.2;       // [uM^-1 s^-1] kon = koff/Kd
    const fp k2Boff = k0Boff / (fp)100.0;     // [s^-1]
    const fp k2Bon  = k0Bon;                  // [uM^-1 s^-1]
    const fp k4Bon  = k0Bon;                  // [uM^-1 s^-1]
    const fp kSLmyo = (fp)8.587e-15;          // [L/msec]
    const fp scale  = (fp)1e-3;

    // compute frequently used base indices
    const int offDyad = initvalu_offset_Dyad;
    const int offSL   = initvalu_offset_SL;
    const int offCyt  = initvalu_offset_Cyt;
    const int offECC  = initvalu_offset_ecc;

    //=====================================================================
    //	COMPUTATION
    //=====================================================================

    // ADJUST ECC incorporate Ca buffering from CaM, convert JCa* from uM/msec to mM/msec
    finavalu[offECC + 35] =
        finavalu[offECC + 35] + scale * JCaDyad;
    finavalu[offECC + 36] =
        finavalu[offECC + 36] + scale * JCaSL;
    finavalu[offECC + 37] =
        finavalu[offECC + 37] + scale * JCaCyt;

    // load Dyad and SL/Cyt species into registers for reuse
    const fp Dyad0  = initvalu[offDyad + 0];
    const fp Dyad1  = initvalu[offDyad + 1];
    const fp Dyad2  = initvalu[offDyad + 2];
    const fp Dyad3  = initvalu[offDyad + 3];
    const fp Dyad4  = initvalu[offDyad + 4];
    const fp Dyad5  = initvalu[offDyad + 5];
    const fp Dyad6  = initvalu[offDyad + 6];
    const fp Dyad7  = initvalu[offDyad + 7];
    const fp Dyad8  = initvalu[offDyad + 8];
    const fp Dyad9  = initvalu[offDyad + 9];
    const fp Dyad12 = initvalu[offDyad + 12];
    const fp Dyad13 = initvalu[offDyad + 13];
    const fp Dyad14 = initvalu[offDyad + 14];

    const fp SL0   = initvalu[offSL + 0];
    const fp SL1   = initvalu[offSL + 1];
    const fp SL2   = initvalu[offSL + 2];

    const fp Cyt0  = initvalu[offCyt + 0];
    const fp Cyt1  = initvalu[offCyt + 1];
    const fp Cyt2  = initvalu[offCyt + 2];

    // incorporate CaM diffusion between compartments
    fp CaMtotDyad =
        Dyad0 + Dyad1 + Dyad2 + Dyad3 + Dyad4 + Dyad5 +
        CaMKIItotDyad * (Dyad6 + Dyad7 + Dyad8 + Dyad9) +
        Dyad12 + Dyad13 + Dyad14;

    const fp Bdyad = BtotDyad - CaMtotDyad; // [uM dyad]

    // Dyad <-> SL fluxes (scaled 1e-3)
    const fp J_cam_dyadSL =
        scale * (k0Boff * Dyad0 - k0Bon * Bdyad * SL0);  // [uM/msec dyad]
    const fp J_ca2cam_dyadSL =
        scale * (k2Boff * Dyad1 - k2Bon * Bdyad * SL1);  // [uM/msec dyad]
    const fp J_ca4cam_dyadSL =
        scale * (k2Boff * Dyad2 - k4Bon * Bdyad * SL2);  // [uM/msec dyad]

    // SL <-> myo fluxes (already [umol/msec])
    const fp J_cam_SLmyo   = kSLmyo * (SL0 - Cyt0);
    const fp J_ca2cam_SLmyo = kSLmyo * (SL1 - Cyt1);
    const fp J_ca4cam_SLmyo = kSLmyo * (SL2 - Cyt2);

    // ADJUST CAM Dyad
    finavalu[offDyad + 0] =
        finavalu[offDyad + 0] - J_cam_dyadSL;
    finavalu[offDyad + 1] =
        finavalu[offDyad + 1] - J_ca2cam_dyadSL;
    finavalu[offDyad + 2] =
        finavalu[offDyad + 2] - J_ca4cam_dyadSL;

    // ADJUST CAM SL
    finavalu[offSL + 0] =
        finavalu[offSL + 0] +
        J_cam_dyadSL * Vdyad_over_VSL -
        J_cam_SLmyo * invVSL;
    finavalu[offSL + 1] =
        finavalu[offSL + 1] +
        J_ca2cam_dyadSL * Vdyad_over_VSL -
        J_ca2cam_SLmyo * invVSL;
    finavalu[offSL + 2] =
        finavalu[offSL + 2] +
        J_ca4cam_dyadSL * Vdyad_over_VSL -
        J_ca4cam_SLmyo * invVSL;

    // ADJUST CAM Cyt
    finavalu[offCyt + 0] =
        finavalu[offCyt + 0] + J_cam_SLmyo * invVmyo;
    finavalu[offCyt + 1] =
        finavalu[offCyt + 1] + J_ca2cam_SLmyo * invVmyo;
    finavalu[offCyt + 2] =
        finavalu[offCyt + 2] + J_ca4cam_SLmyo * invVmyo;
}
