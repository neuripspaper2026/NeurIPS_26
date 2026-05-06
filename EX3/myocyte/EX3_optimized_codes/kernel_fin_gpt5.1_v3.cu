__device__ void kernel_fin(fp * __restrict__ initvalu, int initvalu_offset_ecc, int initvalu_offset_Dyad,
                int initvalu_offset_SL, int initvalu_offset_Cyt, fp * __restrict__ parameter,
                fp * __restrict__ finavalu, fp JCaDyad, fp JCaSL, fp JCaCyt) {

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
    BtotDyad      = parameter[2];      //
    CaMKIItotDyad = parameter[3]; //

    // set variables
    Vmyo  = (fp)2.1454e-11;  // [L]
    Vdyad = (fp)1.7790e-14; // [L]
    VSL   = (fp)6.6013e-13;   // [L]
    kSLmyo = (fp)8.587e-15;    // [L/msec]
    k0Boff = (fp)0.0014;       // [s^-1]
    k0Bon  = k0Boff / (fp)0.2;  // [uM^-1 s^-1] kon = koff/Kd
    k2Boff = k0Boff * (fp)0.01; // [s^-1]
    k2Bon  = k0Bon;         // [uM^-1 s^-1]
    k4Bon  = k0Bon; // [uM^-1 s^-1]

    const fp scale_mM = (fp)1e-3;

    // ADJUST ECC incorporate Ca buffering from CaM, convert JCaCyt from uM/msec
    // to mM/msec
    int ecc_35 = initvalu_offset_ecc + 35;
    int ecc_36 = initvalu_offset_ecc + 36;
    int ecc_37 = initvalu_offset_ecc + 37;

    finavalu[ecc_35] =
        finavalu[ecc_35] + scale_mM * JCaDyad;
    finavalu[ecc_36] =
        finavalu[ecc_36] + scale_mM * JCaSL;
    finavalu[ecc_37] =
        finavalu[ecc_37] + scale_mM * JCaCyt;

    // base offsets
    int dyad0 = initvalu_offset_Dyad;
    int sl0   = initvalu_offset_SL;
    int cyt0  = initvalu_offset_Cyt;

    // load dyad values into registers
    fp dyad_0  = initvalu[dyad0 + 0];
    fp dyad_1  = initvalu[dyad0 + 1];
    fp dyad_2  = initvalu[dyad0 + 2];
    fp dyad_3  = initvalu[dyad0 + 3];
    fp dyad_4  = initvalu[dyad0 + 4];
    fp dyad_5  = initvalu[dyad0 + 5];
    fp dyad_6  = initvalu[dyad0 + 6];
    fp dyad_7  = initvalu[dyad0 + 7];
    fp dyad_8  = initvalu[dyad0 + 8];
    fp dyad_9  = initvalu[dyad0 + 9];
    fp dyad_12 = initvalu[dyad0 + 12];
    fp dyad_13 = initvalu[dyad0 + 13];
    fp dyad_14 = initvalu[dyad0 + 14];

    // incorporate CaM diffusion between compartments
    fp CaMKII_block =
        CaMKIItotDyad * (dyad_6 + dyad_7 + dyad_8 + dyad_9);

    CaMtotDyad = dyad_0 +
                 dyad_1 +
                 dyad_2 +
                 dyad_3 +
                 dyad_4 +
                 dyad_5 +
                 CaMKII_block +
                 dyad_12 +
                 dyad_13 +
                 dyad_14;

    Bdyad = BtotDyad - CaMtotDyad; // [uM dyad]

    // load SL values
    fp sl_0 = initvalu[sl0 + 0];
    fp sl_1 = initvalu[sl0 + 1];
    fp sl_2 = initvalu[sl0 + 2];

    const fp scale_uM = (fp)1e-3;

    J_cam_dyadSL =
        scale_uM *
        (k0Boff * dyad_0 -
         k0Bon * Bdyad * sl_0); // [uM/msec dyad]
    J_ca2cam_dyadSL =
        scale_uM *
        (k2Boff * dyad_1 -
         k2Bon * Bdyad * sl_1); // [uM/msec dyad]
    J_ca4cam_dyadSL =
        scale_uM *
        (k2Boff * dyad_2 -
         k4Bon * Bdyad * sl_2); // [uM/msec dyad]

    // load Cyt values
    fp cyt_0 = initvalu[cyt0 + 0];
    fp cyt_1 = initvalu[cyt0 + 1];
    fp cyt_2 = initvalu[cyt0 + 2];

    // SL-myo fluxes
    fp sl_minus_cyt_0 = sl_0 - cyt_0;
    fp sl_minus_cyt_1 = sl_1 - cyt_1;
    fp sl_minus_cyt_2 = sl_2 - cyt_2;

    J_cam_SLmyo = kSLmyo * sl_minus_cyt_0; // [umol/msec]
    J_ca2cam_SLmyo =
        kSLmyo * sl_minus_cyt_1; // [umol/msec]
    J_ca4cam_SLmyo =
        kSLmyo * sl_minus_cyt_2; // [umol/msec]

    // ADJUST CAM Dyad
    int f_dyad0 = initvalu_offset_Dyad;
    finavalu[f_dyad0 + 0] =
        finavalu[f_dyad0 + 0] - J_cam_dyadSL;
    finavalu[f_dyad0 + 1] =
        finavalu[f_dyad0 + 1] - J_ca2cam_dyadSL;
    finavalu[f_dyad0 + 2] =
        finavalu[f_dyad0 + 2] - J_ca4cam_dyadSL;

    // volume ratios and inverses
    fp Vdyad_over_VSL = Vdyad / VSL;
    fp invVSL         = (fp)1.0 / VSL;
    fp invVmyo        = (fp)1.0 / Vmyo;

    // ADJUST CAM Sl
    int f_sl0 = initvalu_offset_SL;

    fp J_cam_dyadSL_Vdyad_over_VSL =
        J_cam_dyadSL * Vdyad_over_VSL;
    fp J_ca2cam_dyadSL_Vdyad_over_VSL =
        J_ca2cam_dyadSL * Vdyad_over_VSL;
    fp J_ca4cam_dyadSL_Vdyad_over_VSL =
        J_ca4cam_dyadSL * Vdyad_over_VSL;

    fp J_cam_SLmyo_over_VSL =
        J_cam_SLmyo * invVSL;
    fp J_ca2cam_SLmyo_over_VSL =
        J_ca2cam_SLmyo * invVSL;
    fp J_ca4cam_SLmyo_over_VSL =
        J_ca4cam_SLmyo * invVSL;

    finavalu[f_sl0 + 0] = finavalu[f_sl0 + 0] +
                          J_cam_dyadSL_Vdyad_over_VSL -
                          J_cam_SLmyo_over_VSL;
    finavalu[f_sl0 + 1] = finavalu[f_sl0 + 1] +
                          J_ca2cam_dyadSL_Vdyad_over_VSL -
                          J_ca2cam_SLmyo_over_VSL;
    finavalu[f_sl0 + 2] = finavalu[f_sl0 + 2] +
                          J_ca4cam_dyadSL_Vdyad_over_VSL -
                          J_ca4cam_SLmyo_over_VSL;

    // ADJUST CAM Cyt
    int f_cyt0 = initvalu_offset_Cyt;

    fp J_cam_SLmyo_over_Vmyo =
        J_cam_SLmyo * invVmyo;
    fp J_ca2cam_SLmyo_over_Vmyo =
        J_ca2cam_SLmyo * invVmyo;
    fp J_ca4cam_SLmyo_over_Vmyo =
        J_ca4cam_SLmyo * invVmyo;

    finavalu[f_cyt0 + 0] =
        finavalu[f_cyt0 + 0] + J_cam_SLmyo_over_Vmyo;
    finavalu[f_cyt0 + 1] =
        finavalu[f_cyt0 + 1] + J_ca2cam_SLmyo_over_Vmyo;
    finavalu[f_cyt0 + 2] =
        finavalu[f_cyt0 + 2] + J_ca4cam_SLmyo_over_Vmyo;
}
