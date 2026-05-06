__device__ void kernel_fin(fp *initvalu, int initvalu_offset_ecc, int initvalu_offset_Dyad,
                int initvalu_offset_SL, int initvalu_offset_Cyt, fp *parameter,
                fp *finavalu, fp JCaDyad, fp JCaSL, fp JCaCyt) {

    //=====================================================================
    //	VARIABLES
    //=====================================================================

    // Load parameters once from global memory
    const fp BtotDyad = parameter[2];
    const fp CaMKIItotDyad = parameter[3];

    // Precompute constants
    const fp Vmyo = 2.1454e-11;
    const fp Vdyad = 1.7790e-14;
    const fp VSL = 6.6013e-13;
    const fp kSLmyo = 8.587e-15;
    const fp k0Boff = 0.0014;
    const fp k0Bon = k0Boff / 0.2;
    const fp k2Boff = k0Boff / 100;
    const fp k2Bon = k0Bon;
    const fp k4Bon = k0Bon;

    // Precompute scaling factors
    const fp scale_1e3 = 1e-3;
    const fp scale_Vdyad_VSL = Vdyad / VSL;
    const fp scale_inv_VSL = 1.0 / VSL;
    const fp scale_inv_Vmyo = 1.0 / Vmyo;

    // Load initvalu values once from global memory - Dyad compartment
    fp initvalu_Dyad_0 = initvalu[initvalu_offset_Dyad + 0];
    fp initvalu_Dyad_1 = initvalu[initvalu_offset_Dyad + 1];
    fp initvalu_Dyad_2 = initvalu[initvalu_offset_Dyad + 2];
    fp initvalu_Dyad_3 = initvalu[initvalu_offset_Dyad + 3];
    fp initvalu_Dyad_4 = initvalu[initvalu_offset_Dyad + 4];
    fp initvalu_Dyad_5 = initvalu[initvalu_offset_Dyad + 5];
    fp initvalu_Dyad_6 = initvalu[initvalu_offset_Dyad + 6];
    fp initvalu_Dyad_7 = initvalu[initvalu_offset_Dyad + 7];
    fp initvalu_Dyad_8 = initvalu[initvalu_offset_Dyad + 8];
    fp initvalu_Dyad_9 = initvalu[initvalu_offset_Dyad + 9];
    fp initvalu_Dyad_12 = initvalu[initvalu_offset_Dyad + 12];
    fp initvalu_Dyad_13 = initvalu[initvalu_offset_Dyad + 13];
    fp initvalu_Dyad_14 = initvalu[initvalu_offset_Dyad + 14];

    // Load initvalu values once from global memory - SL compartment
    fp initvalu_SL_0 = initvalu[initvalu_offset_SL + 0];
    fp initvalu_SL_1 = initvalu[initvalu_offset_SL + 1];
    fp initvalu_SL_2 = initvalu[initvalu_offset_SL + 2];

    // Load initvalu values once from global memory - Cyt compartment
    fp initvalu_Cyt_0 = initvalu[initvalu_offset_Cyt + 0];
    fp initvalu_Cyt_1 = initvalu[initvalu_offset_Cyt + 1];
    fp initvalu_Cyt_2 = initvalu[initvalu_offset_Cyt + 2];

    // ADJUST ECC incorporate Ca buffering from CaM, convert JCaCyt from uM/msec to mM/msec
    fp finavalu_ecc_35 = finavalu[initvalu_offset_ecc + 35] + scale_1e3 * JCaDyad;
    fp finavalu_ecc_36 = finavalu[initvalu_offset_ecc + 36] + scale_1e3 * JCaSL;
    fp finavalu_ecc_37 = finavalu[initvalu_offset_ecc + 37] + scale_1e3 * JCaCyt;

    // Compute CaMtotDyad - fused computation
    fp CaMtotDyad = initvalu_Dyad_0 + initvalu_Dyad_1 + initvalu_Dyad_2 + 
                    initvalu_Dyad_3 + initvalu_Dyad_4 + initvalu_Dyad_5 +
                    CaMKIItotDyad * (initvalu_Dyad_6 + initvalu_Dyad_7 + 
                                     initvalu_Dyad_8 + initvalu_Dyad_9) +
                    initvalu_Dyad_12 + initvalu_Dyad_13 + initvalu_Dyad_14;

    fp Bdyad = BtotDyad - CaMtotDyad;

    // Compute J fluxes - fused operations
    fp J_cam_dyadSL = scale_1e3 * (k0Boff * initvalu_Dyad_0 - k0Bon * Bdyad * initvalu_SL_0);
    fp J_ca2cam_dyadSL = scale_1e3 * (k2Boff * initvalu_Dyad_1 - k2Bon * Bdyad * initvalu_SL_1);
    fp J_ca4cam_dyadSL = scale_1e3 * (k2Boff * initvalu_Dyad_2 - k4Bon * Bdyad * initvalu_SL_2);

    fp J_cam_SLmyo = kSLmyo * (initvalu_SL_0 - initvalu_Cyt_0);
    fp J_ca2cam_SLmyo = kSLmyo * (initvalu_SL_1 - initvalu_Cyt_1);
    fp J_ca4cam_SLmyo = kSLmyo * (initvalu_SL_2 - initvalu_Cyt_2);

    // Precompute common terms
    fp J_cam_dyadSL_scaled = J_cam_dyadSL * scale_Vdyad_VSL;
    fp J_ca2cam_dyadSL_scaled = J_ca2cam_dyadSL * scale_Vdyad_VSL;
    fp J_ca4cam_dyadSL_scaled = J_ca4cam_dyadSL * scale_Vdyad_VSL;

    fp J_cam_SLmyo_scaled = J_cam_SLmyo * scale_inv_VSL;
    fp J_ca2cam_SLmyo_scaled = J_ca2cam_SLmyo * scale_inv_VSL;
    fp J_ca4cam_SLmyo_scaled = J_ca4cam_SLmyo * scale_inv_VSL;

    // ADJUST CAM Dyad - direct write
    finavalu[initvalu_offset_Dyad + 0] = finavalu[initvalu_offset_Dyad + 0] - J_cam_dyadSL;
    finavalu[initvalu_offset_Dyad + 1] = finavalu[initvalu_offset_Dyad + 1] - J_ca2cam_dyadSL;
    finavalu[initvalu_offset_Dyad + 2] = finavalu[initvalu_offset_Dyad + 2] - J_ca4cam_dyadSL;

    // ADJUST CAM SL - direct write with fused operations
    finavalu[initvalu_offset_SL + 0] = finavalu[initvalu_offset_SL + 0] + J_cam_dyadSL_scaled - J_cam_SLmyo_scaled;
    finavalu[initvalu_offset_SL + 1] = finavalu[initvalu_offset_SL + 1] + J_ca2cam_dyadSL_scaled - J_ca2cam_SLmyo_scaled;
    finavalu[initvalu_offset_SL + 2] = finavalu[initvalu_offset_SL + 2] + J_ca4cam_dyadSL_scaled - J_ca4cam_SLmyo_scaled;

    // ADJUST CAM Cyt - direct write
    finavalu[initvalu_offset_Cyt + 0] = finavalu[initvalu_offset_Cyt + 0] + J_cam_SLmyo * scale_inv_Vmyo;
    finavalu[initvalu_offset_Cyt + 1] = finavalu[initvalu_offset_Cyt + 1] + J_ca2cam_SLmyo * scale_inv_Vmyo;
    finavalu[initvalu_offset_Cyt + 2] = finavalu[initvalu_offset_Cyt + 2] + J_ca4cam_SLmyo * scale_inv_Vmyo;

    // Write ECC adjustments
    finavalu[initvalu_offset_ecc + 35] = finavalu_ecc_35;
    finavalu[initvalu_offset_ecc + 36] = finavalu_ecc_36;
    finavalu[initvalu_offset_ecc + 37] = finavalu_ecc_37;
}
