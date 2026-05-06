__device__ __forceinline__ void kernel_fin(fp * __restrict__ initvalu, int initvalu_offset_ecc,
                                           int initvalu_offset_Dyad, int initvalu_offset_SL,
                                           int initvalu_offset_Cyt, fp * __restrict__ parameter,
                                           fp * __restrict__ finavalu, fp JCaDyad,
                                           fp JCaSL, fp JCaCyt) {

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
    // fp kDyadSL;																			//
    // [L/msec]
    fp kSLmyo; // [L/msec]
    fp k0Boff; // [s^-1]
    fp k0Bon;  // [uM^-1 s^-1] kon = koff/Kd
    fp k2Boff; // [s^-1]
    fp k2Bon;  // [uM^-1 s^-1]
    // fp k4Boff;																			//
    // [s^-1]
    fp k4Bon; // [uM^-1 s^-1]
    fp CaMtotDyad;
    fp Bdyad;           // [uM dyad]
    fp J_cam_dyadSL;    // [uM/msec dyad]
    fp J_ca2cam_dyadSL; // [uM/msec dyad]
    fp J_ca4cam_dyadSL; // [uM/msec dyad]
    fp J_cam_SLmyo;     // [umol/msec]
    fp J_ca2cam_SLmyo;  // [umol/msec]
    fp J_ca4cam_SLmyo;  // [umol/msec]

    // common constants
    const fp inv_1000 = (fp)1e-3;

    //=====================================================================
    //	COMPUTATION
    //=====================================================================

    // decoded input parameters
    BtotDyad      = parameter[2]; //
    CaMKIItotDyad = parameter[3]; //

    // set variables (compile-time constants kept as literals for FMA opportunities)
    Vmyo  = (fp)2.1454e-11;  // [L]
    Vdyad = (fp)1.7790e-14;  // [L]
    VSL   = (fp)6.6013e-13;  // [L]
    // kDyadSL = 3.6363e-16;																//
    // [L/msec]
    kSLmyo = (fp)8.587e-15;     // [L/msec]
    k0Boff = (fp)0.0014;        // [s^-1]
    k0Bon  = k0Boff / (fp)0.2;  // [uM^-1 s^-1] kon = koff/Kd
    k2Boff = k0Boff / (fp)100.0; // [s^-1]
    k2Bon  = k0Bon;             // [uM^-1 s^-1]
    // k4Boff = k2Boff;																	//
    // [s^-1]
    k4Bon = k0Bon; // [uM^-1 s^-1]

    // ADJUST ECC incorporate Ca buffering from CaM, convert JCa* from uM/msec to mM/msec
    const int ecc35 = initvalu_offset_ecc + 35;
    const int ecc36 = initvalu_offset_ecc + 36;
    const int ecc37 = initvalu_offset_ecc + 37;

    finavalu[ecc35] = finavalu[ecc35] + inv_1000 * JCaDyad;
    finavalu[ecc36] = finavalu[ecc36] + inv_1000 * JCaSL;
    finavalu[ecc37] = finavalu[ecc37] + inv_1000 * JCaCyt;

    // incorporate CaM diffusion between compartments
    const int offDy = initvalu_offset_Dyad;
    const int offSL = initvalu_offset_SL;

    const fp dy0  = initvalu[offDy + 0];
    const fp dy1  = initvalu[offDy + 1];
    const fp dy2  = initvalu[offDy + 2];
    const fp dy3  = initvalu[offDy + 3];
    const fp dy4  = initvalu[offDy + 4];
    const fp dy5  = initvalu[offDy + 5];
    const fp dy6  = initvalu[offDy + 6];
    const fp dy7  = initvalu[offDy + 7];
    const fp dy8  = initvalu[offDy + 8];
    const fp dy9  = initvalu[offDy + 9];
    const fp dy12 = initvalu[offDy + 12];
    const fp dy13 = initvalu[offDy + 13];
    const fp dy14 = initvalu[offDy + 14];

    CaMtotDyad =
        dy0 + dy1 + dy2 + dy3 + dy4 + dy5 +
        CaMKIItotDyad * (dy6 + dy7 + dy8 + dy9) +
        dy12 + dy13 + dy14;

    Bdyad = BtotDyad - CaMtotDyad; // [uM dyad]

    const fp sl0 = initvalu[offSL + 0];
    const fp sl1 = initvalu[offSL + 1];
    const fp sl2 = initvalu[offSL + 2];

    J_cam_dyadSL =
        inv_1000 *
        (k0Boff * dy0 - k0Bon * Bdyad * sl0); // [uM/msec dyad]
    J_ca2cam_dyadSL =
        inv_1000 *
        (k2Boff * dy1 - k2Bon * Bdyad * sl1); // [uM/msec dyad]
    J_ca4cam_dyadSL =
        inv_1000 *
        (k2Boff * dy2 - k4Bon * Bdyad * sl2); // [uM/msec dyad]

    const int offCyt = initvalu_offset_Cyt;
    const fp cyt0 = initvalu[offCyt + 0];
    const fp cyt1 = initvalu[offCyt + 1];
    const fp cyt2 = initvalu[offCyt + 2];

    const fp sl0_minus_cyt0 = sl0 - cyt0;
    const fp sl1_minus_cyt1 = sl1 - cyt1;
    const fp sl2_minus_cyt2 = sl2 - cyt2;

    J_cam_SLmyo    = kSLmyo * sl0_minus_cyt0; // [umol/msec]
    J_ca2cam_SLmyo = kSLmyo * sl1_minus_cyt1; // [umol/msec]
    J_ca4cam_SLmyo = kSLmyo * sl2_minus_cyt2; // [umol/msec]

    // ADJUST CAM Dyad
    const int fDy0 = offDy + 0;
    const int fDy1 = offDy + 1;
    const int fDy2 = offDy + 2;

    finavalu[fDy0] = finavalu[fDy0] - J_cam_dyadSL;
    finavalu[fDy1] = finavalu[fDy1] - J_ca2cam_dyadSL;
    finavalu[fDy2] = finavalu[fDy2] - J_ca4cam_dyadSL;

    // precompute ratios
    const fp Vdyad_over_VSL = Vdyad / VSL;
    const fp inv_VSL        = (fp)1.0 / VSL;
    const fp inv_Vmyo       = (fp)1.0 / Vmyo;

    // ADJUST CAM Sl
    const int fSL0 = offSL + 0;
    const int fSL1 = offSL + 1;
    const int fSL2 = offSL + 2;

    finavalu[fSL0] = finavalu[fSL0] +
                     J_cam_dyadSL * Vdyad_over_VSL -
                     J_cam_SLmyo * inv_VSL;
    finavalu[fSL1] = finavalu[fSL1] +
                     J_ca2cam_dyadSL * Vdyad_over_VSL -
                     J_ca2cam_SLmyo * inv_VSL;
    finavalu[fSL2] = finavalu[fSL2] +
                     J_ca4cam_dyadSL * Vdyad_over_VSL -
                     J_ca4cam_SLmyo * inv_VSL;

    // ADJUST CAM Cyt
    const int fCyt0 = offCyt + 0;
    const int fCyt1 = offCyt + 1;
    const int fCyt2 = offCyt + 2;

    finavalu[fCyt0] = finavalu[fCyt0] + J_cam_SLmyo * inv_Vmyo;
    finavalu[fCyt1] = finavalu[fCyt1] + J_ca2cam_SLmyo * inv_Vmyo;
    finavalu[fCyt2] = finavalu[fCyt2] + J_ca4cam_SLmyo * inv_Vmyo;
}
