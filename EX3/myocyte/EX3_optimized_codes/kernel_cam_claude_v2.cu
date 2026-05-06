__device__ void kernel_cam(fp timeinst, fp *d_initvalu, fp *d_finavalu,
                           int valu_offset, fp *d_params, int params_offset,
                           fp *d_com, int com_offset, fp Ca) {

    // Load all input values from global memory once
    const fp Btot = d_params[params_offset + 1];
    const fp CaMKIItot = d_params[params_offset + 2];
    const fp CaNtot = d_params[params_offset + 3];
    const fp PP1tot = d_params[params_offset + 4];
    const fp K = d_params[16];
    const fp Mg = d_params[17];

    // Load state variables with coalesced access pattern
    const fp CaM = d_initvalu[valu_offset];
    const fp Ca2CaM = d_initvalu[valu_offset + 1];
    const fp Ca4CaM = d_initvalu[valu_offset + 2];
    const fp CaMB = d_initvalu[valu_offset + 3];
    const fp Ca2CaMB = d_initvalu[valu_offset + 4];
    const fp Ca4CaMB = d_initvalu[valu_offset + 5];
    const fp Pb2 = d_initvalu[valu_offset + 6];
    const fp Pb = d_initvalu[valu_offset + 7];
    const fp Pt = d_initvalu[valu_offset + 8];
    const fp Pt2 = d_initvalu[valu_offset + 9];
    const fp Pa = d_initvalu[valu_offset + 10];
    const fp Ca4CaN = d_initvalu[valu_offset + 11];
    const fp CaMCa4CaN = d_initvalu[valu_offset + 12];
    const fp Ca2CaMCa4CaN = d_initvalu[valu_offset + 13];
    const fp Ca4CaMCa4CaN = d_initvalu[valu_offset + 14];

    // Precompute Ca^2 once
    const fp Ca2 = Ca * Ca;

    // Ca/CaM parameters - optimized branching
    fp Kd02, Kd24;
    const fp Mg_clamped = fminf(Mg, 1.0f);
    const fp Mg_excess = Mg - 1.0f;
    
    if (Mg <= 1.0f) {
        Kd02 = 0.0025f * (1.0f + K / 0.94f - Mg / 0.012f) *
               (1.0f + K / 8.1f + Mg / 0.022f);
        Kd24 = 0.128f * (1.0f + K / 0.64f + Mg / 0.0014f) *
               (1.0f + K / 13.0f - Mg / 0.153f);
    } else {
        Kd02 = 0.0025f * (1.0f + K / 0.94f - 1.0f / 0.012f + Mg_excess / 0.060f) *
               (1.0f + K / 8.1f + 1.0f / 0.022f + Mg_excess / 0.068f);
        Kd24 = 0.128f * (1.0f + K / 0.64f + 1.0f / 0.0014f + Mg_excess / 0.005f) *
               (1.0f + K / 13.0f - 1.0f / 0.153f + Mg_excess / 0.150f);
    }
    
    const fp k20 = 10.0f;
    const fp k02 = k20 / Kd02;
    const fp k42 = 500.0f;
    const fp k24 = k42 / Kd24;

    // CaM buffering parameters
    const fp k0Boff = 0.0014f;
    const fp k0Bon = k0Boff / 0.2f;
    const fp k2Boff = k0Boff / 100.0f;
    const fp k2Bon = k0Bon;
    const fp k4Boff = k2Boff;
    const fp k4Bon = k0Bon;

    const fp k20B = k20 / 100.0f;
    const fp k02B = k02;
    const fp k42B = k42;
    const fp k24B = k24;

    // Wi Wa Wt Wp parameters
    const fp kbi = 2.2f;
    const fp kib = kbi / 33.5e-3f;
    const fp kpp1 = 1.72f;
    const fp Kmpp1 = 11.5f;
    const fp kib2 = kib;
    const fp kb2i = kib2 * 5.0f;
    const fp kb24 = k24;
    const fp kb42 = k42 * 33.5e-3f / 5.0f;
    const fp kta = kbi / 1000.0f;
    const fp kat = kib;
    const fp kt42 = k42 * 33.5e-6f / 5.0f;
    const fp kt24 = k24;
    const fp kat2 = kib;
    const fp kt2a = kib * 5.0f;

    // CaN parameters
    const fp kcanCaoff = 1.0f;
    const fp kcanCaon = kcanCaoff / 0.5f;
    const fp kcanCaM4on = 46.0f;
    const fp kcanCaM4off = 0.0013f;
    const fp kcanCaM2on = kcanCaM4on;
    const fp kcanCaM2off = 2508.0f * kcanCaM4off;
    const fp kcanCaM0on = kcanCaM4on;
    const fp kcanCaM0off = 165.0f * kcanCaM2off;
    const fp k02can = k02;
    const fp k20can = k20 / 165.0f;
    const fp k24can = k24;
    const fp k42can = k20 / 2508.0f;

    // CaM Reaction fluxes - use precomputed Ca2
    const fp rcn02 = k02 * Ca2 * CaM - k20 * Ca2CaM;
    const fp rcn24 = k24 * Ca2 * Ca2CaM - k42 * Ca4CaM;

    // CaM buffer fluxes
    const fp B = Btot - CaMB - Ca2CaMB - Ca4CaMB;
    const fp rcn02B = k02B * Ca2 * CaMB - k20B * Ca2CaMB;
    const fp rcn24B = k24B * Ca2 * Ca2CaMB - k42B * Ca4CaMB;
    const fp rcn0B = k0Bon * CaM * B - k0Boff * CaMB;
    const fp rcn2B = k2Bon * Ca2CaM * B - k2Boff * Ca2CaMB;
    const fp rcn4B = k4Bon * Ca4CaM * B - k4Boff * Ca4CaMB;

    // CaN reaction fluxes
    const fp Ca2CaN = CaNtot - Ca4CaN - CaMCa4CaN - Ca2CaMCa4CaN - Ca4CaMCa4CaN;
    const fp rcnCa4CaN = kcanCaon * Ca2 * Ca2CaN - kcanCaoff * Ca4CaN;
    const fp rcn02CaN = k02can * Ca2 * CaMCa4CaN - k20can * Ca2CaMCa4CaN;
    const fp rcn24CaN = k24can * Ca2 * Ca2CaMCa4CaN - k42can * Ca4CaMCa4CaN;
    const fp rcn0CaN = kcanCaM0on * CaM * Ca4CaN - kcanCaM0off * CaMCa4CaN;
    const fp rcn2CaN = kcanCaM2on * Ca2CaM * Ca4CaN - kcanCaM2off * Ca2CaMCa4CaN;
    const fp rcn4CaN = kcanCaM4on * Ca4CaM * Ca4CaN - kcanCaM4off * Ca4CaMCa4CaN;

    // CaMKII reaction fluxes
    const fp Pix = 1.0f - Pb2 - Pb - Pt - Pt2 - Pa;
    const fp rcnCKib2 = kib2 * Ca2CaM * Pix - kb2i * Pb2;
    const fp rcnCKb2b = kb24 * Ca2 * Pb2 - kb42 * Pb;
    const fp rcnCKib = kib * Ca4CaM * Pix - kbi * Pb;
    const fp T = Pb + Pt + Pt2 + Pa;
    const fp T2 = T * T;
    const fp kbt = fmaf(0.015f, T2 * T, fmaf(0.0074f, T2, 0.055f * T));
    const fp denom_inv_Pt = 1.0f / (Kmpp1 + CaMKIItot * Pt);
    const fp rcnCKbt = kbt * Pb - kpp1 * PP1tot * Pt * denom_inv_Pt;
    const fp rcnCKtt2 = kt42 * Pt - kt24 * Ca2 * Pt2;
    const fp rcnCKta = kta * Pt - kat * Ca4CaM * Pa;
    const fp rcnCKt2a = kt2a * Pt2 - kat2 * Ca2CaM * Pa;
    const fp denom_inv_Pt2 = 1.0f / (Kmpp1 + CaMKIItot * Pt2);
    const fp rcnCKt2b2 = kpp1 * PP1tot * Pt2 * denom_inv_Pt2;
    const fp denom_inv_Pa = 1.0f / (Kmpp1 + CaMKIItot * Pa);
    const fp rcnCKai = kpp1 * PP1tot * Pa * denom_inv_Pa;

    // Compute derivatives with fused multiply-add
    const fp scale = 1e-3f;
    
    const fp dCaM = scale * (-rcn02 - rcn0B - rcn0CaN);
    const fp dCa2CaM = scale * (rcn02 - rcn24 - rcn2B - rcn2CaN +
                                CaMKIItot * (-rcnCKib2 + rcnCKt2a));
    const fp dCa4CaM = scale * (rcn24 - rcn4B - rcn4CaN + 
                                CaMKIItot * (-rcnCKib + rcnCKta));
    const fp dCaMB = scale * (rcn0B - rcn02B);
    const fp dCa2CaMB = scale * (rcn02B + rcn2B - rcn24B);
    const fp dCa4CaMB = scale * (rcn24B + rcn4B);

    const fp dPb2 = scale * (rcnCKib2 - rcnCKb2b + rcnCKt2b2);
    const fp dPb = scale * (rcnCKib + rcnCKb2b - rcnCKbt);
    const fp dPt = scale * (rcnCKbt - rcnCKta - rcnCKtt2);
    const fp dPt2 = scale * (rcnCKtt2 - rcnCKt2a - rcnCKt2b2);
    const fp dPa = scale * (rcnCKta + rcnCKt2a - rcnCKai);

    const fp dCa4CaN_val = scale * (rcnCa4CaN - rcn0CaN - rcn2CaN - rcn4CaN);
    const fp dCaMCa4CaN_val = scale * (rcn0CaN - rcn02CaN);
    const fp dCa2CaMCa4CaN_val = scale * (rcn2CaN + rcn02CaN - rcn24CaN);
    const fp dCa4CaMCa4CaN_val = scale * (rcn4CaN + rcn24CaN);

    // Store results with coalesced writes
    d_finavalu[valu_offset] = dCaM;
    d_finavalu[valu_offset + 1] = dCa2CaM;
    d_finavalu[valu_offset + 2] = dCa4CaM;
    d_finavalu[valu_offset + 3] = dCaMB;
    d_finavalu[valu_offset + 4] = dCa2CaMB;
    d_finavalu[valu_offset + 5] = dCa4CaMB;
    d_finavalu[valu_offset + 6] = dPb2;
    d_finavalu[valu_offset + 7] = dPb;
    d_finavalu[valu_offset + 8] = dPt;
    d_finavalu[valu_offset + 9] = dPt2;
    d_finavalu[valu_offset + 10] = dPa;
    d_finavalu[valu_offset + 11] = dCa4CaN_val;
    d_finavalu[valu_offset + 12] = dCaMCa4CaN_val;
    d_finavalu[valu_offset + 13] = dCa2CaMCa4CaN_val;
    d_finavalu[valu_offset + 14] = dCa4CaMCa4CaN_val;

    // Compute and store communication offset value
    const fp JCa = scale * (2.0f * CaMKIItot * (rcnCKtt2 - rcnCKb2b) -
                            2.0f * (rcn02 + rcn24 + rcn02B + rcn24B + 
                                    rcnCa4CaN + rcn02CaN + rcn24CaN));
    d_finavalu[com_offset] = JCa;
}
