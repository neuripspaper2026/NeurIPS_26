<<<CODE>>>
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

    // decoding input array - use registers for frequently accessed values
    fp CaM;
    fp Ca2CaM;
    fp Ca4CaM;
    fp CaMB;
    fp Ca2CaMB;
    fp Ca4CaMB;
    fp Pb2;
    fp Pb;
    fp Pt;
    fp Pt2;
    fp Pa;
    fp Ca4CaN;
    fp CaMCa4CaN;
    fp Ca2CaMCa4CaN;
    fp Ca4CaMCa4CaN;

    // Ca/CaM parameters
    fp Kd02;
    fp Kd24;
    fp k20;
    fp k02;
    fp k42;
    fp k24;

    // CaM buffering (B) parameters
    fp k0Boff;
    fp k0Bon;
    fp k2Boff;
    fp k2Bon;
    fp k4Boff;
    fp k4Bon;

    // using thermodynamic constraints
    fp k20B;
    fp k02B;
    fp k42B;
    fp k24B;

    // Wi Wa Wt Wp
    fp kbi;
    fp kib;
    fp kpp1;
    fp Kmpp1;
    fp kib2;
    fp kb2i;
    fp kb24;
    fp kb42;
    fp kta;
    fp kat;
    fp kt42;
    fp kt24;
    fp kat2;
    fp kt2a;

    // CaN parameters
    fp kcanCaoff;
    fp kcanCaon;
    fp kcanCaM4on;
    fp kcanCaM4off;
    fp kcanCaM2on;
    fp kcanCaM2off;
    fp kcanCaM0on;
    fp kcanCaM0off;
    fp k02can;
    fp k20can;
    fp k24can;
    fp k42can;

    // CaM Reaction fluxes
    fp rcn02;
    fp rcn24;

    // CaM buffer fluxes
    fp B;
    fp rcn02B;
    fp rcn24B;
    fp rcn0B;
    fp rcn2B;
    fp rcn4B;

    // CaN reaction fluxes
    fp Ca2CaN;
    fp rcnCa4CaN;
    fp rcn02CaN;
    fp rcn24CaN;
    fp rcn0CaN;
    fp rcn2CaN;
    fp rcn4CaN;

    // CaMKII reaction fluxes
    fp Pix;
    fp rcnCKib2;
    fp rcnCKb2b;
    fp rcnCKib;
    fp T;
    fp kbt;
    fp rcnCKbt;
    fp rcnCKtt2;
    fp rcnCKta;
    fp rcnCKt2a;
    fp rcnCKt2b2;
    fp rcnCKai;

    // CaM equations
    fp dCaM;
    fp dCa2CaM;
    fp dCa4CaM;
    fp dCaMB;
    fp dCa2CaMB;
    fp dCa4CaMB;

    // CaMKII equations
    fp dPb2;
    fp dPb;
    fp dPt;
    fp dPt2;
    fp dPa;

    // CaN equations
    fp dCa4CaN;
    fp dCaMCa4CaN;
    fp dCa2CaMCa4CaN;
    fp dCa4CaMCa4CaN;

    // Precompute commonly used values
    fp Ca_squared;
    fp Ca_pow4;

    //=====================================================================
    //	EXECUTION
    //=====================================================================

    // Load inputs from global memory once
    Btot = d_params[params_offset + 1];
    CaMKIItot = d_params[params_offset + 2];
    CaNtot = d_params[params_offset + 3];
    PP1tot = d_params[params_offset + 4];
    K = d_params[16];
    Mg = d_params[17];

    // Precompute Ca powers
    Ca_squared = Ca * Ca;
    Ca_pow4 = Ca_squared * Ca_squared;

    // Load initial values from global memory in coalesced manner
    CaM = d_initvalu[valu_offset];
    Ca2CaM = d_initvalu[valu_offset + 1];
    Ca4CaM = d_initvalu[valu_offset + 2];
    CaMB = d_initvalu[valu_offset + 3];
    Ca2CaMB = d_initvalu[valu_offset + 4];
    Ca4CaMB = d_initvalu[valu_offset + 5];
    Pb2 = d_initvalu[valu_offset + 6];
    Pb = d_initvalu[valu_offset + 7];
    Pt = d_initvalu[valu_offset + 8];
    Pt2 = d_initvalu[valu_offset + 9];
    Pa = d_initvalu[valu_offset + 10];
    Ca4CaN = d_initvalu[valu_offset + 11];
    CaMCa4CaN = d_initvalu[valu_offset + 12];
    Ca2CaMCa4CaN = d_initvalu[valu_offset + 13];
    Ca4CaMCa4CaN = d_initvalu[valu_offset + 14];

    // Ca/CaM parameters - optimized computation
    fp K_term1 = K / 0.94;
    fp K_term2 = K / 8.1;
    fp K_term3 = K / 0.64;
    fp K_term4 = K / 13.0;
    
    if (Mg <= 1) {
        Kd02 = 0.0025 * (1 + K_term1 - Mg / 0.012) * (1 + K_term2 + Mg / 0.022);
        Kd24 = 0.128 * (1 + K_term3 + Mg / 0.0014) * (1 + K_term4 - Mg / 0.153);
    } else {
        fp Mg_minus_1 = Mg - 1;
        Kd02 = 0.0025 * (1 + K_term1 - 1 / 0.012 + Mg_minus_1 / 0.060) *
               (1 + K_term2 + 1 / 0.022 + Mg_minus_1 / 0.068);
        Kd24 = 0.128 * (1 + K_term3 + 1 / 0.0014 + Mg_minus_1 / 0.005) *
               (1 + K_term4 - 1 / 0.153 + Mg_minus_1 / 0.150);
    }
    
    k20 = 10;
    k02 = k20 / Kd02;
    k42 = 500;
    k24 = k42 / Kd24;

    // CaM buffering (B) parameters
    k0Boff = 0.0014;
    k0Bon = k0Boff / 0.2;
    k2Boff = k0Boff / 100;
    k2Bon = k0Bon;
    k4Boff = k2Boff;
    k4Bon = k0Bon;

    // using thermodynamic constraints
    k20B = k20 / 100;
    k02B = k02;
    k42B = k42;
    k24B = k24;

    // Wi Wa Wt Wp
    kbi = 2.2;
    kib = kbi / 33.5e-3;
    kpp1 = 1.72;
    Kmpp1 = 11.5;
    kib2 = kib;
    kb2i = kib2 * 5;
    kb24 = k24;
    kb42 = k42 * 33.5e-3 / 5;
    kta = kbi / 1000;
    kat = kib;
    kt42 = k42 * 33.5e-6 / 5;
    kt24 = k24;
    kat2 = kib;
    kt2a = kib * 5;

    // CaN parameters
    kcanCaoff = 1;
    kcanCaon = kcanCaoff / 0.5;
    kcanCaM4on = 46;
    kcanCaM4off = 0.0013;
    kcanCaM2on = kcanCaM4on;
    kcanCaM2off = 2508 * kcanCaM4off;
    kcanCaM0on = kcanCaM4on;
    kcanCaM0off = 165 * kcanCaM2off;
    k02can = k02;
    k20can = k20 / 165;
    k24can = k24;
    k42can = k20 / 2508;

    // CaM Reaction fluxes - use precomputed Ca_squared
    rcn02 = k02 * Ca_squared * CaM - k20 * Ca2CaM;
    rcn24 = k24 * Ca_squared * Ca2CaM - k42 * Ca4CaM;

    // CaM buffer fluxes
    B = Btot - CaMB - Ca2CaMB - Ca4CaMB;
    rcn02B = k02B * Ca_squared * CaMB - k20B * Ca2CaMB;
    rcn24B = k24B * Ca_squared * Ca2CaMB - k42B * Ca4CaMB;
    rcn0B = k0Bon * CaM * B - k0Boff * CaMB;
    rcn2B = k2Bon * Ca2CaM * B - k2Boff * Ca2CaMB;
    rcn4B = k4Bon * Ca4CaM * B - k4Boff * Ca4CaMB;

    // CaN reaction fluxes
    Ca2CaN = CaNtot - Ca4CaN - CaMCa4CaN - Ca2CaMCa4CaN - Ca4CaMCa4CaN;
    rcnCa4CaN = kcanCaon * Ca_squared * Ca2CaN - kcanCaoff * Ca4CaN;
    rcn02CaN = k02can * Ca_squared * CaMCa4CaN - k20can * Ca2CaMCa4CaN;
    rcn24CaN = k24can * Ca_squared * Ca2CaMCa4CaN - k42can * Ca4CaMCa4CaN;
    rcn0CaN = kcanCaM0on * CaM * Ca4CaN - kcanCaM0off * CaMCa4CaN;
    rcn2CaN = kcanCaM2on * Ca2CaM * Ca4CaN - kcanCaM2off * Ca2CaMCa4CaN;
    rcn4CaN = kcanCaM4on * Ca4CaM * Ca4CaN - kcanCaM4off * Ca4CaMCa4CaN;

    // CaMKII reaction fluxes
    Pix = 1 - Pb2 - Pb - Pt - Pt2 - Pa;
    rcnCKib2 = kib2 * Ca2CaM * Pix - kb2i * Pb2;
    rcnCKb2b = kb24 * Ca_squared * Pb2 - kb42 * Pb;
    rcnCKib = kib * Ca4CaM * Pix - kbi * Pb;
    T = Pb + Pt + Pt2 + Pa;
    fp T_squared = T * T;
    fp T_cubed = T_squared * T;
    kbt = 0.055 * T + 0.0074 * T_squared + 0.015 * T_cubed;
    
    fp CaMKIItot_Pt = CaMKIItot * Pt;
    fp CaMKIItot_Pt2 = CaMKIItot * Pt2;
    fp CaMKIItot_Pa = CaMKIItot * Pa;
    fp kpp1_PP1tot = kpp1 * PP1tot;
    
    rcnCKbt = kbt * Pb - kpp1_PP1tot * Pt / (Kmpp1 + CaMKIItot_Pt);
    rcnCKtt2 = kt42 * Pt - kt24 * Ca_squared * Pt2;
    rcnCKta = kta * Pt - kat * Ca4CaM * Pa;
    rcnCKt2a = kt2a * Pt2 - kat2 * Ca2CaM * Pa;
    rcnCKt2b2 = kpp1_PP1tot * Pt2 / (Kmpp1 + CaMKIItot_Pt2);
    rcnCKai = kpp1_PP1tot * Pa / (Kmpp1 + CaMKIItot_Pa);

    // CaM equations
    fp scale_factor = 1e-3;
    dCaM = scale_factor * (-rcn02 - rcn0B - rcn0CaN);
    dCa2CaM = scale_factor * (rcn02 - rcn24 - rcn2B - rcn2CaN + CaMKIItot * (-rcnCKib2 + rcnCKt2a));
    dCa4CaM = scale_factor * (rcn24 - rcn4B - rcn4CaN + CaMKIItot * (-rcnCKib + rcnCKta));
    dCaMB = scale_factor * (rcn0B - rcn02B);
    dCa2CaMB = scale_factor * (rcn02B + rcn2B - rcn24B);
    dCa4CaMB = scale_factor * (rcn24B + rcn4B);

    // CaMKII equations
    dPb2 = scale_factor * (rcnCKib2 - rcnCKb2b + rcnCKt2b2);
    dPb = scale_factor * (rcnCKib + rcnCKb2b - rcnCKbt);
    dPt = scale_factor * (rcnCKbt - rcnCKta - rcnCKtt2);
    dPt2 = scale_factor * (rcnCKtt2 - rcnCKt2a - rcnCKt2b2);
    dPa = scale_factor * (rcnCKta + rcnCKt2a - rcnCKai);

    // CaN equations
    dCa4CaN = scale_factor * (rcnCa4CaN - rcn0CaN - rcn2CaN - rcn4CaN);
    dCaMCa
