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
    BtotDyad      = parameter[2]; //
    CaMKIItotDyad = parameter[3]; //

    // set variables (hoisted constants)
    Vmyo   = (fp)2.1454e-11;  // [L]
    Vdyad  = (fp)1.7790e-14;  // [L]
    VSL    = (fp)6.6013e-13;  // [L]
    kSLmyo = (fp)8.587e-15;   // [L/msec]
    k0Boff = (fp)0.0014;      // [s^-1]
    k0Bon  = k0Boff / (fp)0.2;// [uM^-1 s^-1] kon = koff/Kd
    k2Boff = k0Boff / (fp)100;// [s^-1]
    k2Bon  = k0Bon;           // [uM^-1 s^-1]
    k4Bon  = k0Bon;           // [uM^-1 s^-1]

    const fp scale_mM = (fp)1e-3;

    // ADJUST ECC incorporate Ca buffering from CaM, convert JCaCyt from uM/msec
    // to mM/msec
    const int ecc35 = initvalu_offset_ecc + 35;
    const int ecc36 = initvalu_offset_ecc + 36;
    const int ecc37 = initvalu_offset_ecc + 37;
    finavalu[ecc35] = finavalu[ecc35] + scale_mM * JCaDyad;
    finavalu[ecc36] = finavalu[ecc36] + scale_mM * JCaSL;
    finavalu[ecc37] = finavalu[ecc37] + scale_mM * JCaCyt;

    // precompute commonly used indices
    const int dy0  = initvalu_offset_Dyad + 0;
    const int dy1  = initvalu_offset_Dyad + 1;
    const int dy2  = initvalu_offset_Dyad + 2;
    const int dy3  = initvalu_offset_Dyad + 3;
    const int dy4  = initvalu_offset_Dyad + 4;
    const int dy5  = initvalu_offset_Dyad + 5;
    const int dy6  = initvalu_offset_Dyad + 6;
    const int dy7  = initvalu_offset_Dyad + 7;
    const int dy8  = initvalu_offset_Dyad + 8;
    const int dy9  = initvalu_offset_Dyad + 9;
    const int dy12 = initvalu_offset_Dyad + 12;
    const int dy13 = initvalu_offset_Dyad + 13;
    const int dy14 = initvalu_offset_Dyad + 14;

    const int sl0 = initvalu_offset_SL + 0;
    const int sl1 = initvalu_offset_SL + 1;
    const int sl2 = initvalu_offset_SL + 2;

    const int cy0 = initvalu_offset_Cyt + 0;
    const int cy1 = initvalu_offset_Cyt + 1;
    const int cy2 = initvalu_offset_Cyt + 2;

    // load Dyad values once into registers for reuse
    const fp dyad0  = initvalu[dy0];
    const fp dyad1  = initvalu[dy1];
    const fp dyad2  = initvalu[dy2];
    const fp dyad3  = initvalu[dy3];
    const fp dyad4  = initvalu[dy4];
    const fp dyad5  = initvalu[dy5];
    const fp dyad6  = initvalu[dy6];
    const fp dyad7  = initvalu[dy7];
    const fp dyad8  = initvalu[dy8];
    const fp dyad9  = initvalu[dy9];
    const fp dyad12 = initvalu[dy12];
    const fp dyad13 = initvalu[dy13];
    const fp dyad14 = initvalu[dy14];

    // incorporate CaM diffusion between compartments
    CaMtotDyad = dyad0 + dyad1 + dyad2 + dyad3 + dyad4 + dyad5 +
                 CaMKIItotDyad * (dyad6 + dyad7 + dyad8 + dyad9) +
                 dyad12 + dyad13 + dyad14;
    Bdyad = BtotDyad - CaMtotDyad; // [uM dyad]

    const fp sl_0 = initvalu[sl0];
    const fp sl_1 = initvalu[sl1];
    const fp sl_2 = initvalu[sl2];

    // diffusion fluxes between Dyad and SL (scaled once by 1e-3)
    J_cam_dyadSL =
        scale_mM *
        (k0Boff * dyad0 -
         k0Bon * Bdyad * sl_0); // [uM/msec dyad]
    J_ca2cam_dyadSL =
        scale_mM *
        (k2Boff * dyad1 -
         k2Bon * Bdyad * sl_1); // [uM/msec dyad]
    J_ca4cam_dyadSL =
        scale_mM *
        (k2Boff * dyad2 -
         k4Bon * Bdyad * sl_2); // [uM/msec dyad]

    const fp cyt_0 = initvalu[cy0];
    const fp cyt_1 = initvalu[cy1];
    const fp cyt_2 = initvalu[cy2];

    // diffusion fluxes between SL and myoplasm
    const fp sl_minus_cy0 = sl_0 - cyt_0;
    const fp sl_minus_cy1 = sl_1 - cyt_1;
    const fp sl_minus_cy2 = sl_2 - cyt_2;

    J_cam_SLmyo   = kSLmyo * sl_minus_cy0; // [umol/msec]
    J_ca2cam_SLmyo= kSLmyo * sl_minus_cy1; // [umol/msec]
    J_ca4cam_SLmyo= kSLmyo * sl_minus_cy2; // [umol/msec]

    const fp Vdyad_over_VSL = Vdyad / VSL;
    const fp invVSL         = (fp)1 / VSL;
    const fp invVmyo        = (fp)1 / Vmyo;

    // ADJUST CAM Dyad
    finavalu[dy0] = finavalu[dy0] - J_cam_dyadSL;
    finavalu[dy1] = finavalu[dy1] - J_ca2cam_dyadSL;
    finavalu[dy2] = finavalu[dy2] - J_ca4cam_dyadSL;

    // ADJUST CAM SL
    finavalu[sl0] = finavalu[sl0] +
                    J_cam_dyadSL * Vdyad_over_VSL -
                    J_cam_SLmyo * invVSL;
    finavalu[sl1] = finavalu[sl1] +
                    J_ca2cam_dyadSL * Vdyad_over_VSL -
                    J_ca2cam_SLmyo * invVSL;
    finavalu[sl2] = finavalu[sl2] +
                    J_ca4cam_dyadSL * Vdyad_over_VSL -
                    J_ca4cam_SLmyo * invVSL;

    // ADJUST CAM Cyt
    finavalu[cy0] = finavalu[cy0] + J_cam_SLmyo * invVmyo;
    finavalu[cy1] = finavalu[cy1] + J_ca2cam_SLmyo * invVmyo;
    finavalu[cy2] = finavalu[cy2] + J_ca4cam_SLmyo * invVmyo;
}
