__device__ void kernel_cam(fp timeinst, fp *d_initvalu, fp *d_finavalu,
                           int valu_offset, fp *d_params, int params_offset,
                           fp *d_com, int com_offset, fp Ca) {

    //=====================================================================
    //	VARIABLES
    //=====================================================================

    // inputs
    fp Btot;
    fp CaMKIItot;
    fp CaNtot;
    fp PP1tot;
    fp K;
    fp Mg;

    // decoding input array - load once from global memory
    fp CaM = d_initvalu[valu_offset];
    fp Ca2CaM = d_initvalu[valu_offset + 1];
    fp Ca4CaM = d_initvalu[valu_offset + 2];
    fp CaMB = d_initvalu[valu_offset + 3];
    fp Ca2CaMB = d_initvalu[valu_offset + 4];
    fp Ca4CaMB = d_initvalu[valu_offset + 5];
    fp Pb2 = d_initvalu[valu_offset + 6];
    fp Pb = d_initvalu[valu_offset + 7];
    fp Pt = d_initvalu[valu_offset + 8];
    fp Pt2 = d_initvalu[valu_offset + 9];
    fp Pa = d_initvalu[valu_offset + 10];
    fp Ca4CaN = d_initvalu[valu_offset + 11];
    fp CaMCa4CaN = d_initvalu[valu_offset + 12];
    fp Ca2CaMCa4CaN = d_initvalu[valu_offset + 13];
    fp Ca4CaMCa4CaN = d_initvalu[valu_offset + 14];

    // Load parameters once
    Btot = d_params[params_offset + 1];
    CaMKIItot = d_params[params_offset + 2];
    CaNtot = d_params[params_offset + 3];
    PP1tot = d_params[params_offset + 4];
    K = d_params[16];
    Mg = d_params[17];

    // Ca/CaM parameters - compute once
    fp Kd02, Kd24;
    if (Mg <= 1) {
        fp term1_02 = 1 + K / 0.94 - Mg / 0.012;
        fp term2_02 = 1 + K / 8.1 + Mg / 0.022;
        Kd02 = 0.0025 * term1_02 * term2_02;
        
        fp term1_24 = 1 + K / 0.64 + Mg / 0.0014;
        fp term2_24 = 1 + K / 13.0 - Mg / 0.153;
        Kd24 = 0.128 * term1_24 * term2_24;
    } else {
        fp Mg_adj = Mg - 1;
        fp term1_02 = 1 + K / 0.94 - 1 / 0.012 + Mg_adj / 0.060;
        fp term2_02 = 1 + K / 8.1 + 1 / 0.022 + Mg_adj / 0.068;
        Kd02 = 0.0025 * term1_02 * term2_02;
        
        fp term1_24 = 1 + K / 0.64 + 1 / 0.0014 + Mg_adj / 0.005;
        fp term2_24 = 1 + K / 13.0 - 1 / 0.153 + Mg_adj / 0.150;
        Kd24 = 0.128 * term1_24 * term2_24;
    }
    
    const fp k20 = 10;
    const fp k02 = k20 / Kd02;
    const fp k42 = 500;
    const fp k24 = k42 / Kd24;

    // CaM buffering (B) parameters
    const fp k0Boff = 0.0014;
    const fp k0Bon = k0Boff / 0.2;
    const fp k2Boff = k0Boff / 100;
    const fp k2Bon = k0Bon;
    const fp k4Boff = k2Boff;
    const fp k4Bon = k0Bon;

    // using thermodynamic constraints
    const fp k20B = k20 / 100;
    const fp k02B = k02;
    const fp k42B = k42;
    const fp k24B = k24;

    // Wi Wa Wt Wp
    const fp kbi = 2.2;
    const fp kib = kbi / 33.5e-3;
    const fp kpp1 = 1.72;
    const fp Kmpp1 = 11.5;
    const fp kib2 = kib;
    const fp kb2i = kib2 * 5;
    const fp kb24 = k24;
    const fp kb42 = k42 * 33.5e-3 / 5;
    const fp kta = kbi / 1000;
    const fp kat = kib;
    const fp kt42 = k42 * 33.5e-6 / 5;
    const fp kt24 = k24;
    const fp kat2 = kib;
    const fp kt2a = kib * 5;

    // CaN parameters
    const fp kcanCaoff = 1;
    const fp kcanCaon = kcanCaoff / 0.5;
    const fp kcanCaM4on = 46;
    const fp kcanCaM4off = 0.0013;
    const fp kcanCaM2on = kcanCaM4on;
    const fp kcanCaM2off = 2508 * kcanCaM4off;
    const fp kcanCaM0on = kcanCaM4on;
    const fp kcanCaM0off = 165 * kcanCaM2off;
    const fp k02can = k02;
    const fp k20can = k20 / 165;
    const fp k24can = k24;
    const fp k42can = k20 / 2508;

    // Precompute Ca^2 once
    const fp Ca2 = Ca * Ca;

    // CaM Reaction fluxes
    fp rcn02 = k02 * Ca2 * CaM - k20 * Ca2CaM;
    fp rcn24 = k24 * Ca2 * Ca2CaM - k42 * Ca4CaM;

    // CaM buffer fluxes
    fp B = Btot - CaMB - Ca2CaMB - Ca4CaMB;
    fp rcn02B = k02B * Ca2 * CaMB - k20B * Ca2CaMB;
    fp rcn24B = k24B * Ca2 * Ca2CaMB - k42B * Ca4CaMB;
    fp rcn0B = k0Bon * CaM * B - k0Boff * CaMB;
    fp rcn2B = k2Bon * Ca2CaM * B - k2Boff * Ca2CaMB;
    fp rcn4B = k4Bon * Ca4CaM * B - k4Boff * Ca4CaMB;

    // CaN reaction fluxes
    fp Ca2CaN = CaNtot - Ca4CaN - CaMCa4CaN - Ca2CaMCa4CaN - Ca4CaMCa4CaN;
    fp rcnCa4CaN = kcanCaon * Ca2 * Ca2CaN - kcanCaoff * Ca4CaN;
    fp rcn02CaN = k02can * Ca2 * CaMCa4CaN - k20can * Ca2CaMCa4CaN;
    fp rcn24CaN = k24can * Ca2 * Ca2CaMCa4CaN - k42can * Ca4CaMCa4CaN;
    fp rcn0CaN = kcanCaM0on * CaM * Ca4CaN - kcanCaM0off * CaMCa4CaN;
    fp rcn2CaN = kcanCaM2on * Ca2CaM * Ca4CaN - kcanCaM2off * Ca2CaMCa4CaN;
    fp rcn4CaN = kcanCaM4on * Ca4CaM * Ca4CaN - kcanCaM4off * Ca4CaMCa4CaN;

    // CaMKII reaction fluxes
    fp Pix = 1 - Pb2 - Pb - Pt - Pt2 - Pa;
    fp rcnCKib2 = kib2 * Ca2CaM * Pix - kb2i * Pb2;
    fp rcnCKb2b = kb24 * Ca2 * Pb2 - kb42 * Pb;
    fp rcnCKib = kib * Ca4CaM * Pix - kbi * Pb;
    fp T = Pb + Pt + Pt2 + Pa;
    fp T2 = T * T;
    fp kbt = 0.055 * T + 0.0074 * T2 + 0.015 * T2 * T;
    fp denom_Pt = Kmpp1 + CaMKIItot * Pt;
    fp rcnCKbt = kbt * Pb - kpp1 * PP1tot * Pt / denom_Pt;
    fp rcnCKtt2 = kt42 * Pt - kt24 * Ca2 * Pt2;
    fp rcnCKta = kta * Pt - kat * Ca4CaM * Pa;
    fp rcnCKt2a = kt2a * Pt2 - kat2 * Ca2CaM * Pa;
    fp denom_Pt2 = Kmpp1 + CaMKIItot * Pt2;
    fp rcnCKt2b2 = kpp1 * PP1tot * Pt2 / denom_Pt2;
    fp denom_Pa = Kmpp1 + CaMKIItot * Pa;
    fp rcnCKai = kpp1 * PP1tot * Pa / denom_Pa;

    // CaM equations - compute and store directly
    const fp scale = 1e-3;
    d_finavalu[valu_offset] = scale * (-rcn02 - rcn0B - rcn0CaN);
    d_finavalu[valu_offset + 1] = scale * (rcn02 - rcn24 - rcn2B - rcn2CaN + CaMKIItot * (-rcnCKib2 + rcnCKt2a));
    d_finavalu[valu_offset + 2] = scale * (rcn24 - rcn4B - rcn4CaN + CaMKIItot * (-rcnCKib + rcnCKta));
    d_finavalu[valu_offset + 3] = scale * (rcn0B - rcn02B);
    d_finavalu[valu_offset + 4] = scale * (rcn02B + rcn2B - rcn24B);
    d_finavalu[valu_offset + 5] = scale * (rcn24B + rcn4B);

    // CaMKII equations
    d_finavalu[valu_offset + 6] = scale * (rcnCKib2 - rcnCKb2b + rcnCKt2b2);
    d_finavalu[valu_offset + 7] = scale * (rcnCKib + rcnCKb2b - rcnCKbt);
    d_finavalu[valu_offset + 8] = scale * (rcnCKbt - rcnCKta - rcnCKtt2);
    d_finavalu[valu_offset + 9] = scale * (rcnCKtt2 - rcnCKt2a - rcnCKt2b2);
    d_finavalu[valu_offset + 10] = scale * (rcnCKta + rcnCKt2a - rcnCKai);

    // CaN equations
    d_finavalu[valu_offset + 11] = scale * (rcnCa4CaN - rcn0CaN - rcn2CaN - rcn4CaN);
    d_finavalu[valu_offset + 12] = scale * (rcn0CaN - rcn02CaN);
    d_finavalu[valu_offset + 13] = scale * (rcn2CaN + rcn02CaN - rcn24CaN);
    d_finavalu[valu_offset + 14] = scale * (rcn4CaN + rcn24CaN);

    // write to global variables for adjusting Ca buffering in EC coupling model
    d_finavalu[com_offset] = scale * (2 * CaMKIItot * (rcnCKtt2 - rcnCKb2b) -
                2 * (rcn02 + rcn24 + rcn02B + rcn24B + rcnCa4CaN + rcn02CaN + rcn24CaN));
}
