__device__ void kernel_fin(fp * __restrict__ initvalu,
                const int initvalu_offset_ecc,
                const int initvalu_offset_Dyad,
                const int initvalu_offset_SL,
                const int initvalu_offset_Cyt,
                fp * __restrict__ parameter,
                fp * __restrict__ finavalu,
                const fp JCaDyad,
                const fp JCaSL,
                const fp JCaCyt) {

    //=====================================================================
    //	VARIABLES
    //=====================================================================

    // decoded input parameters
    const fp BtotDyad      = parameter[2];
    const fp CaMKIItotDyad = parameter[3];

    // compute variables (constants hoisted as const for better register usage)
    const fp Vmyo  = (fp)2.1454e-11;  // [L]
    const fp Vdyad = (fp)1.7790e-14;  // [L]
    const fp VSL   = (fp)6.6013e-13;  // [L]

    const fp kSLmyo = (fp)8.587e-15;    // [L/msec]
    const fp k0Boff = (fp)0.0014;       // [s^-1]
    const fp k0Bon  = k0Boff / (fp)0.2; // [uM^-1 s^-1] kon = koff/Kd
    const fp k2Boff = k0Boff / (fp)100.0; // [s^-1]
    const fp k2Bon  = k0Bon;              // [uM^-1 s^-1]
    const fp k4Bon  = k0Bon;              // [uM^-1 s^-1]

    const fp scale_1em3 = (fp)1e-3;

    //=====================================================================
    //	COMPUTATION
    //=====================================================================

    // ADJUST ECC incorporate Ca buffering from CaM, convert JCaCyt from uM/msec
    // to mM/msec
    const int ecc_35 = initvalu_offset_ecc + 35;
    const int ecc_36 = initvalu_offset_ecc + 36;
    const int ecc_37 = initvalu_offset_ecc + 37;

    finavalu[ecc_35] = finavalu[ecc_35] + scale_1em3 * JCaDyad;
    finavalu[ecc_36] = finavalu[ecc_36] + scale_1em3 * JCaSL;
    finavalu[ecc_37] = finavalu[ecc_37] + scale_1em3 * JCaCyt;

    // precompute frequently used offsets
    const int Dyad_0  = initvalu_offset_Dyad + 0;
    const int Dyad_1  = initvalu_offset_Dyad + 1;
    const int Dyad_2  = initvalu_offset_Dyad + 2;
    const int Dyad_3  = initvalu_offset_Dyad + 3;
    const int Dyad_4  = initvalu_offset_Dyad + 4;
    const int Dyad_5  = initvalu_offset_Dyad + 5;
    const int Dyad_6  = initvalu_offset_Dyad + 6;
    const int Dyad_7  = initvalu_offset_Dyad + 7;
    const int Dyad_8  = initvalu_offset_Dyad + 8;
    const int Dyad_9  = initvalu_offset_Dyad + 9;
    const int Dyad_12 = initvalu_offset_Dyad + 12;
    const int Dyad_13 = initvalu_offset_Dyad + 13;
    const int Dyad_14 = initvalu_offset_Dyad + 14;

    const int SL_0 = initvalu_offset_SL + 0;
    const int SL_1 = initvalu_offset_SL + 1;
    const int SL_2 = initvalu_offset_SL + 2;

    const int Cyt_0 = initvalu_offset_Cyt + 0;
    const int Cyt_1 = initvalu_offset_Cyt + 1;
    const int Cyt_2 = initvalu_offset_Cyt + 2;

    // load dyad species into registers
    const fp Dyad_0_v  = initvalu[Dyad_0];
    const fp Dyad_1_v  = initvalu[Dyad_1];
    const fp Dyad_2_v  = initvalu[Dyad_2];
    const fp Dyad_3_v  = initvalu[Dyad_3];
    const fp Dyad_4_v  = initvalu[Dyad_4];
    const fp Dyad_5_v  = initvalu[Dyad_5];
    const fp Dyad_6_v  = initvalu[Dyad_6];
    const fp Dyad_7_v  = initvalu[Dyad_7];
    const fp Dyad_8_v  = initvalu[Dyad_8];
    const fp Dyad_9_v  = initvalu[Dyad_9];
    const fp Dyad_12_v = initvalu[Dyad_12];
    const fp Dyad_13_v = initvalu[Dyad_13];
    const fp Dyad_14_v = initvalu[Dyad_14];

    const fp SL_0_v = initvalu[SL_0];
    const fp SL_1_v = initvalu[SL_1];
    const fp SL_2_v = initvalu[SL_2];

    const fp Cyt_0_v = initvalu[Cyt_0];
    const fp Cyt_1_v = initvalu[Cyt_1];
    const fp Cyt_2_v = initvalu[Cyt_2];

    // incorporate CaM diffusion between compartments
    const fp CaMKIItotDyad_scaled =
        CaMKIItotDyad * (Dyad_6_v + Dyad_7_v + Dyad_8_v + Dyad_9_v);

    const fp CaMtotDyad =
        Dyad_0_v + Dyad_1_v + Dyad_2_v + Dyad_3_v + Dyad_4_v + Dyad_5_v +
        CaMKIItotDyad_scaled +
        Dyad_12_v + Dyad_13_v + Dyad_14_v;

    const fp Bdyad = BtotDyad - CaMtotDyad; // [uM dyad]

    const fp tmp_Bdyad_SL0 = Bdyad * SL_0_v;
    const fp tmp_Bdyad_SL1 = Bdyad * SL_1_v;
    const fp tmp_Bdyad_SL2 = Bdyad * SL_2_v;

    const fp J_cam_dyadSL =
        scale_1em3 *
        (k0Boff * Dyad_0_v - k0Bon * tmp_Bdyad_SL0); // [uM/msec dyad]

    const fp J_ca2cam_dyadSL =
        scale_1em3 *
        (k2Boff * Dyad_1_v - k2Bon * tmp_Bdyad_SL1); // [uM/msec dyad]

    const fp J_ca4cam_dyadSL =
        scale_1em3 *
        (k2Boff * Dyad_2_v - k4Bon * tmp_Bdyad_SL2); // [uM/msec dyad]

    const fp diff_SL0_Cyt0 = SL_0_v - Cyt_0_v;
    const fp diff_SL1_Cyt1 = SL_1_v - Cyt_1_v;
    const fp diff_SL2_Cyt2 = SL_2_v - Cyt_2_v;

    const fp J_cam_SLmyo   = kSLmyo * diff_SL0_Cyt0; // [umol/msec]
    const fp J_ca2cam_SLmyo = kSLmyo * diff_SL1_Cyt1; // [umol/msec]
    const fp J_ca4cam_SLmyo = kSLmyo * diff_SL2_Cyt2; // [umol/msec]

    const fp Vdyad_over_VSL = Vdyad / VSL;
    const fp inv_VSL        = (fp)1.0 / VSL;
    const fp inv_Vmyo       = (fp)1.0 / Vmyo;

    // ADJUST CAM Dyad
    finavalu[Dyad_0] -= J_cam_dyadSL;
    finavalu[Dyad_1] -= J_ca2cam_dyadSL;
    finavalu[Dyad_2] -= J_ca4cam_dyadSL;

    // ADJUST CAM SL
    finavalu[SL_0] += J_cam_dyadSL * Vdyad_over_VSL - J_cam_SLmyo * inv_VSL;
    finavalu[SL_1] += J_ca2cam_dyadSL * Vdyad_over_VSL -
                      J_ca2cam_SLmyo * inv_VSL;
    finavalu[SL_2] += J_ca4cam_dyadSL * Vdyad_over_VSL -
                      J_ca4cam_SLmyo * inv_VSL;

    // ADJUST CAM Cyt
    finavalu[Cyt_0] += J_cam_SLmyo * inv_Vmyo;
    finavalu[Cyt_1] += J_ca2cam_SLmyo * inv_Vmyo;
    finavalu[Cyt_2] += J_ca4cam_SLmyo * inv_Vmyo;
}
