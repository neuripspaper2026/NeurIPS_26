__device__ void kernel_fin(fp *initvalu, int initvalu_offset_ecc, int initvalu_offset_Dyad,
                int initvalu_offset_SL, int initvalu_offset_Cyt, fp *parameter,
                fp *finavalu, fp JCaDyad, fp JCaSL, fp JCaCyt) {

    // Load parameters into registers for faster access
    const fp BtotDyad = parameter[2];
    const fp CaMKIItotDyad = parameter[3];

    // Constants - precomputed and stored in registers
    const fp Vmyo = 2.1454e-11;
    const fp Vdyad = 1.7790e-14;
    const fp VSL = 6.6013e-13;
    const fp kSLmyo = 8.587e-15;
    const fp k0Boff = 0.0014;
    const fp k0Bon = 0.007;  // k0Boff / 0.2 precomputed
    const fp k2Boff = 1.4e-5;  // k0Boff / 100 precomputed
    const fp k2Bon = 0.007;  // k0Bon
    const fp k4Bon = 0.007;  // k0Bon
    const fp Vdyad_div_VSL = 2.6941e-2;  // Vdyad / VSL precomputed
    const fp inv_VSL = 1.51499e12;  // 1 / VSL precomputed
    const fp inv_Vmyo = 4.6611e10;  // 1 / Vmyo precomputed
    const fp k0Boff_1e3 = 1.4;  // k0Boff * 1e-3 precomputed
    const fp k2Boff_1e3 = 1.4e-8;  // k2Boff * 1e-3 precomputed

    // Compute ECC offsets
    const int ecc_35 = initvalu_offset_ecc + 35;
    const int ecc_36 = initvalu_offset_ecc + 36;
    const int ecc_37 = initvalu_offset_ecc + 37;

    // Compute Dyad offsets
    const int dyad_0 = initvalu_offset_Dyad;
    const int dyad_1 = initvalu_offset_Dyad + 1;
    const int dyad_2 = initvalu_offset_Dyad + 2;
    const int dyad_3 = initvalu_offset_Dyad + 3;
    const int dyad_4 = initvalu_offset_Dyad + 4;
    const int dyad_5 = initvalu_offset_Dyad + 5;
    const int dyad_6 = initvalu_offset_Dyad + 6;
    const int dyad_7 = initvalu_offset_Dyad + 7;
    const int dyad_8 = initvalu_offset_Dyad + 8;
    const int dyad_9 = initvalu_offset_Dyad + 9;
    const int dyad_12 = initvalu_offset_Dyad + 12;
    const int dyad_13 = initvalu_offset_Dyad + 13;
    const int dyad_14 = initvalu_offset_Dyad + 14;

    // Compute SL offsets
    const int sl_0 = initvalu_offset_SL;
    const int sl_1 = initvalu_offset_SL + 1;
    const int sl_2 = initvalu_offset_SL + 2;

    // Compute Cyt offsets
    const int cyt_0 = initvalu_offset_Cyt;
    const int cyt_1 = initvalu_offset_Cyt + 1;
    const int cyt_2 = initvalu_offset_Cyt + 2;

    // Load initvalu values into registers (coalesced memory access pattern)
    const fp initvalu_dyad_0 = initvalu[dyad_0];
    const fp initvalu_dyad_1 = initvalu[dyad_1];
    const fp initvalu_dyad_2 = initvalu[dyad_2];
    const fp initvalu_dyad_3 = initvalu[dyad_3];
    const fp initvalu_dyad_4 = initvalu[dyad_4];
    const fp initvalu_dyad_5 = initvalu[dyad_5];
    const fp initvalu_dyad_6 = initvalu[dyad_6];
    const fp initvalu_dyad_7 = initvalu[dyad_7];
    const fp initvalu_dyad_8 = initvalu[dyad_8];
    const fp initvalu_dyad_9 = initvalu[dyad_9];
    const fp initvalu_dyad_12 = initvalu[dyad_12];
    const fp initvalu_dyad_13 = initvalu[dyad_13];
    const fp initvalu_dyad_14 = initvalu[dyad_14];
    const fp initvalu_sl_0 = initvalu[sl_0];
    const fp initvalu_sl_1 = initvalu[sl_1];
    const fp initvalu_sl_2 = initvalu[sl_2];
    const fp initvalu_cyt_0 = initvalu[cyt_0];
    const fp initvalu_cyt_1 = initvalu[cyt_1];
    const fp initvalu_cyt_2 = initvalu[cyt_2];

    // ADJUST ECC: incorporate Ca buffering from CaM, convert JCaCyt from uM/msec to mM/msec
    const fp JCaDyad_1e3 = 1e-3 * JCaDyad;
    const fp JCaSL_1e3 = 1e-3 * JCaSL;
    const fp JCaCyt_1e3 = 1e-3 * JCaCyt;

    // Compute CaMtotDyad with fused operations
    const fp CaMKII_sum = CaMKIItotDyad * (initvalu_dyad_6 + initvalu_dyad_7 + 
                                            initvalu_dyad_8 + initvalu_dyad_9);
    const fp CaMtotDyad = initvalu_dyad_0 + initvalu_dyad_1 + initvalu_dyad_2 + 
                          initvalu_dyad_3 + initvalu_dyad_4 + initvalu_dyad_5 + 
                          CaMKII_sum + initvalu_dyad_12 + initvalu_dyad_13 + 
                          initvalu_dyad_14;
    
    const fp Bdyad = BtotDyad - CaMtotDyad;

    // Compute J fluxes with precomputed constants
    const fp J_cam_dyadSL = k0Boff_1e3 * initvalu_dyad_0 - 
                            0.007e-3 * Bdyad * initvalu_sl_0;
    const fp J_ca2cam_dyadSL = k2Boff_1e3 * initvalu_dyad_1 - 
                               0.007e-3 * Bdyad * initvalu_sl_1;
    const fp J_ca4cam_dyadSL = k2Boff_1e3 * initvalu_dyad_2 - 
                               0.007e-3 * Bdyad * initvalu_sl_2;

    // Compute SL-myo fluxes
    const fp diff_sl_cyt_0 = initvalu_sl_0 - initvalu_cyt_0;
    const fp diff_sl_cyt_1 = initvalu_sl_1 - initvalu_cyt_1;
    const fp diff_sl_cyt_2 = initvalu_sl_2 - initvalu_cyt_2;
    
    const fp J_cam_SLmyo = kSLmyo * diff_sl_cyt_0;
    const fp J_ca2cam_SLmyo = kSLmyo * diff_sl_cyt_1;
    const fp J_ca4cam_SLmyo = kSLmyo * diff_sl_cyt_2;

    // Precompute common terms for SL adjustments
    const fp J_cam_dyadSL_scaled = J_cam_dyadSL * Vdyad_div_VSL;
    const fp J_ca2cam_dyadSL_scaled = J_ca2cam_dyadSL * Vdyad_div_VSL;
    const fp J_ca4cam_dyadSL_scaled = J_ca4cam_dyadSL * Vdyad_div_VSL;
    
    const fp J_cam_SLmyo_scaled = J_cam_SLmyo * inv_VSL;
    const fp J_ca2cam_SLmyo_scaled = J_ca2cam_SLmyo * inv_VSL;
    const fp J_ca4cam_SLmyo_scaled = J_ca4cam_SLmyo * inv_VSL;

    // Precompute common terms for Cyt adjustments
    const fp J_cam_SLmyo_cyt = J_cam_SLmyo * inv_Vmyo;
    const fp J_ca2cam_SLmyo_cyt = J_ca2cam_SLmyo * inv_Vmyo;
    const fp J_ca4cam_SLmyo_cyt = J_ca4cam_SLmyo * inv_Vmyo;

    // Load finavalu values and compute updates
    const fp finavalu_ecc_35 = finavalu[ecc_35] + JCaDyad_1e3;
    const fp finavalu_ecc_36 = finavalu[ecc_36] + JCaSL_1e3;
    const fp finavalu_ecc_37 = finavalu[ecc_37] + JCaCyt_1e3;

    const fp finavalu_dyad_0 = finavalu[dyad_0] - J_cam_dyadSL;
    const fp finavalu_dyad_1 = finavalu[dyad_1] - J_ca2cam_dyadSL;
    const fp finavalu_dyad_2 = finavalu[dyad_2] - J_ca4cam_dyadSL;

    const fp finavalu_sl_0 = finavalu[sl_0] + J_cam_dyadSL_scaled - J_cam_SLmyo_scaled;
    const fp finavalu_sl_1 = finavalu[sl_1] + J_ca2cam_dyadSL_scaled - J_ca2cam_SLmyo_scaled;
    const fp finavalu_sl_2 = finavalu[sl_2] + J_ca4cam_dyadSL_scaled - J_ca4cam_SLmyo_scaled;

    const fp finavalu_cyt_0 = finavalu[cyt_0] + J_cam_SLmyo_cyt;
    const fp finavalu_cyt_1 = finavalu[cyt_1] + J_ca2cam_SLmyo_cyt;
    const fp finavalu_cyt_2 = finavalu[cyt_2] + J_ca4cam_SLmyo_cyt;

    // Coalesced writes to global memory
    finavalu[ecc_35] = finavalu_ecc_35;
    finavalu[ecc_36] = finavalu_ecc_36;
    finavalu[ecc_37] = finavalu_ecc_37;

    finavalu[dyad_0] = finavalu_dyad_0;
    finavalu[dyad_1] = finavalu_dyad_1;
    finavalu[dyad_2] = finavalu_dyad_2;

    finavalu[sl_0] = finavalu_sl_0;
    finavalu[sl_1] = finavalu_sl_1;
    finavalu[sl_2] = finavalu_sl_2;

    finavalu[cyt_0] = finavalu_cyt_0;
    finavalu[cyt_1] = finavalu_cyt_1;
    finavalu[cyt_2] = finavalu_cyt_2;
}
