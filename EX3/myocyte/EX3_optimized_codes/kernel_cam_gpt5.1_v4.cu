__device__ void kernel_cam(fp timeinst, fp *d_initvalu, fp *d_finavalu,
                           int valu_offset, fp *d_params, int params_offset,
                           fp *d_com, int com_offset, fp Ca) {

    //=====================================================================
    //  VARIABLES
    //=====================================================================

    // inputs
    fp Btot;
    fp CaMKIItot;
    fp CaNtot;
    fp PP1tot;
    fp K;
    fp Mg;

    // variable references
    const int offset_1  = valu_offset;
    const int offset_2  = valu_offset + 1;
    const int offset_3  = valu_offset + 2;
    const int offset_4  = valu_offset + 3;
    const int offset_5  = valu_offset + 4;
    const int offset_6  = valu_offset + 5;
    const int offset_7  = valu_offset + 6;
    const int offset_8  = valu_offset + 7;
    const int offset_9  = valu_offset + 8;
    const int offset_10 = valu_offset + 9;
    const int offset_11 = valu_offset + 10;
    const int offset_12 = valu_offset + 11;
    const int offset_13 = valu_offset + 12;
    const int offset_14 = valu_offset + 13;
    const int offset_15 = valu_offset + 14;

    // decoding input array
    fp CaM            = d_initvalu[offset_1];
    fp Ca2CaM         = d_initvalu[offset_2];
    fp Ca4CaM         = d_initvalu[offset_3];
    fp CaMB           = d_initvalu[offset_4];
    fp Ca2CaMB        = d_initvalu[offset_5];
    fp Ca4CaMB        = d_initvalu[offset_6];
    fp Pb2            = d_initvalu[offset_7];
    fp Pb             = d_initvalu[offset_8];
    fp Pt             = d_initvalu[offset_9];
    fp Pt2            = d_initvalu[offset_10];
    fp Pa             = d_initvalu[offset_11];
    fp Ca4CaN         = d_initvalu[offset_12];
    fp CaMCa4CaN      = d_initvalu[offset_13];
    fp Ca2CaMCa4CaN   = d_initvalu[offset_14];
    fp Ca4CaMCa4CaN   = d_initvalu[offset_15];

    // common small constants & reuse
    const fp one_thousandth = (fp)1e-3;
    const fp Ca2            = Ca * Ca;

    // inputs
    Btot      = d_params[params_offset + 1];
    CaMKIItot = d_params[params_offset + 2];
    CaNtot    = d_params[params_offset + 3];
    PP1tot    = d_params[params_offset + 4];
    K         = d_params[16];
    Mg        = d_params[17];

    // Ca/CaM parameters
    fp Kd02, Kd24;
    if (Mg <= (fp)1.0) {
        fp t1 = (fp)1.0 + K / (fp)0.94 - Mg / (fp)0.012;
        fp t2 = (fp)1.0 + K / (fp)8.1 + Mg / (fp)0.022;
        fp t3 = (fp)1.0 + K / (fp)0.64 + Mg / (fp)0.0014;
        fp t4 = (fp)1.0 + K / (fp)13.0 - Mg / (fp)0.153;
        Kd02 = (fp)0.0025 * t1 * t2;   // [uM^2]
        Kd24 = (fp)0.128  * t3 * t4;   // [uM^2]
    } else {
        fp Mg_minus_1 = Mg - (fp)1.0;
        fp t1 = (fp)1.0 + K / (fp)0.94 - (fp)1.0 / (fp)0.012 + Mg_minus_1 / (fp)0.060;
        fp t2 = (fp)1.0 + K / (fp)8.1 + (fp)1.0 / (fp)0.022 + Mg_minus_1 / (fp)0.068;
        fp t3 = (fp)1.0 + K / (fp)0.64 + (fp)1.0 / (fp)0.0014 + Mg_minus_1 / (fp)0.005;
        fp t4 = (fp)1.0 + K / (fp)13.0 - (fp)1.0 / (fp)0.153 + Mg_minus_1 / (fp)0.150;
        Kd02 = (fp)0.0025 * t1 * t2;   // [uM^2]
        Kd24 = (fp)0.128  * t3 * t4;   // [uM^2]
    }

    const fp k20 = (fp)10.0;              // [s^-1]
    const fp k02 = k20 / Kd02;            // [uM^-2 s^-1]
    const fp k42 = (fp)500.0;             // [s^-1]
    const fp k24 = k42 / Kd24;            // [uM^-2 s^-1]

    // CaM buffering (B) parameters
    const fp k0Boff = (fp)0.0014;               // [s^-1]
    const fp k0Bon  = k0Boff / (fp)0.2;         // [uM^-1 s^-1] kon = koff/Kd
    const fp k2Boff = k0Boff / (fp)100.0;       // [s^-1]
    const fp k2Bon  = k0Bon;                    // [uM^-1 s^-1]
    const fp k4Boff = k2Boff;                   // [s^-1]
    const fp k4Bon  = k0Bon;                    // [uM^-1 s^-1]

    // using thermodynamic constraints
    const fp k20B = k20 / (fp)100.0;            // [s^-1]
    const fp k02B = k02;                        // [uM^-2 s^-1]
    const fp k42B = k42;                        // [s^-1]
    const fp k24B = k24;                        // [uM^-2 s^-1]

    // Wi Wa Wt Wp
    const fp kbi   = (fp)2.2;                   // [s^-1] (Ca4CaM dissocation from Wb)
    const fp kib   = kbi / (fp)33.5e-3;         // [uM^-1 s^-1]
    const fp kpp1  = (fp)1.72;                  // [s^-1] (PP1-dep dephosphorylation rates)
    const fp Kmpp1 = (fp)11.5;                  // [uM]
    const fp kib2  = kib;
    const fp kb2i  = kib2 * (fp)5.0;
    const fp kb24  = k24;
    const fp kb42  = k42 * (fp)33.5e-3 / (fp)5.0;
    const fp kta   = kbi / (fp)1000.0;          // [s^-1] (Ca4CaM dissociation from Wt)
    const fp kat   = kib;                       // [uM^-1 s^-1] (Ca4CaM reassociation with Wa)
    const fp kt42  = k42 * (fp)33.5e-6 / (fp)5.0;
    const fp kt24  = k24;
    const fp kat2  = kib;
    const fp kt2a  = kib * (fp)5.0;

    // CaN parameters
    const fp kcanCaoff  = (fp)1.0;              // [s^-1]
    const fp kcanCaon   = kcanCaoff / (fp)0.5;  // [uM^-1 s^-1]
    const fp kcanCaM4on = (fp)46.0;             // [uM^-1 s^-1]
    const fp kcanCaM4off= (fp)0.0013;           // [s^-1]
    const fp kcanCaM2on = kcanCaM4on;
    const fp kcanCaM2off= (fp)2508.0 * kcanCaM4off;
    const fp kcanCaM0on = kcanCaM4on;
    const fp kcanCaM0off= (fp)165.0 * kcanCaM2off;
    const fp k02can     = k02;
    const fp k20can     = k20 / (fp)165.0;
    const fp k24can     = k24;
    const fp k42can     = k20 / (fp)2508.0;

    // CaM Reaction fluxes
    const fp rcn02 = k02 * Ca2 * CaM - k20 * Ca2CaM;
    const fp rcn24 = k24 * Ca2 * Ca2CaM - k42 * Ca4CaM;

    // CaM buffer fluxes
    const fp B      = Btot - CaMB - Ca2CaMB - Ca4CaMB;
    const fp rcn02B = k02B * Ca2 * CaMB   - k20B * Ca2CaMB;
    const fp rcn24B = k24B * Ca2 * Ca2CaMB- k42B * Ca4CaMB;
    const fp rcn0B  = k0Bon * CaM    * B  - k0Boff * CaMB;
    const fp rcn2B  = k2Bon * Ca2CaM * B  - k2Boff * Ca2CaMB;
    const fp rcn4B  = k4Bon * Ca4CaM * B  - k4Boff * Ca4CaMB;

    // CaN reaction fluxes
    const fp Ca2CaN   = CaNtot - Ca4CaN - CaMCa4CaN - Ca2CaMCa4CaN - Ca4CaMCa4CaN;
    const fp rcnCa4CaN= kcanCaon   * Ca2 * Ca2CaN      - kcanCaoff  * Ca4CaN;
    const fp rcn02CaN = k02can     * Ca2 * CaMCa4CaN   - k20can     * Ca2CaMCa4CaN;
    const fp rcn24CaN = k24can     * Ca2 * Ca2CaMCa4CaN- k42can     * Ca4CaMCa4CaN;
    const fp rcn0CaN  = kcanCaM0on * CaM  * Ca4CaN     - kcanCaM0off* CaMCa4CaN;
    const fp rcn2CaN  = kcanCaM2on * Ca2CaM * Ca4CaN   - kcanCaM2off* Ca2CaMCa4CaN;
    const fp rcn4CaN  = kcanCaM4on * Ca4CaM * Ca4CaN   - kcanCaM4off* Ca4CaMCa4CaN;

    // CaMKII reaction fluxes
    const fp Pix      = (fp)1.0 - Pb2 - Pb - Pt - Pt2 - Pa;
    const fp rcnCKib2 = kib2 * Ca2CaM * Pix - kb2i * Pb2;
    const fp rcnCKb2b = kb24 * Ca2 * Pb2   - kb42 * Pb;
    const fp rcnCKib  = kib  * Ca4CaM * Pix- kbi  * Pb;
    const fp T        = Pb + Pt + Pt2 + Pa;
    const fp T2       = T * T;
    const fp T3       = T2 * T;
    const fp kbt      = (fp)0.055 * T + (fp)0.0074 * T2 + (fp)0.015 * T3;
    const fp denomPt  = Kmpp1 + CaMKIItot * Pt;
    const fp denomPt2 = Kmpp1 + CaMKIItot * Pt2;
    const fp denomPa  = Kmpp1 + CaMKIItot * Pa;
    const fp rcnCKbt  = kbt * Pb - kpp1 * PP1tot * Pt  / denomPt;
    const fp rcnCKtt2 = kt42 * Pt  - kt24 * Ca2 * Pt2;
    const fp rcnCKta  = kta  * Pt  - kat  * Ca4CaM * Pa;
    const fp rcnCKt2a = kt2a * Pt2 - kat2 * Ca2CaM * Pa;
    const fp rcnCKt2b2= kpp1 * PP1tot * Pt2 / denomPt2;
    const fp rcnCKai  = kpp1 * PP1tot * Pa  / denomPa;

    // CaM equations
    const fp dCaM      = one_thousandth * (-rcn02 - rcn0B - rcn0CaN);
    const fp dCa2CaM   = one_thousandth * (rcn02 - rcn24 - rcn2B - rcn2CaN +
                                           CaMKIItot * (-rcnCKib2 + rcnCKt2a));
    const fp dCa4CaM   = one_thousandth * (rcn24 - rcn4B - rcn4CaN +
                                           CaMKIItot * (-rcnCKib + rcnCKta));
    const fp dCaMB     = one_thousandth * (rcn0B - rcn02B);
    const fp dCa2CaMB  = one_thousandth * (rcn02B + rcn2B - rcn24B);
    const fp dCa4CaMB  = one_thousandth * (rcn24B + rcn4B);

    // CaMKII equations
    const fp dPb2 = one_thousandth * (rcnCKib2 - rcnCKb2b + rcnCKt2b2); // Pb2
    const fp dPb  = one_thousandth * (rcnCKib  + rcnCKb2b - rcnCKbt);   // Pb
    const fp dPt  = one_thousandth * (rcnCKbt  - rcnCKta  - rcnCKtt2);  // Pt
    const fp dPt2 = one_thousandth * (rcnCKtt2 - rcnCKt2a - rcnCKt2b2); // Pt2
    const fp dPa  = one_thousandth * (rcnCKta  + rcnCKt2a - rcnCKai);   // Pa

    // CaN equations
    const fp dCa4CaN       = one_thousandth * (rcnCa4CaN - rcn0CaN - rcn2CaN - rcn4CaN); // Ca4CaN
    const fp dCaMCa4CaN    = one_thousandth * (rcn0CaN   - rcn02CaN);                    // CaMCa4CaN
    const fp dCa2CaMCa4CaN = one_thousandth * (rcn2CaN   + rcn02CaN - rcn24CaN);         // Ca2CaMCa4CaN
    const fp dCa4CaMCa4CaN = one_thousandth * (rcn4CaN   + rcn24CaN);                    // Ca4CaMCa4CaN

    // encode output array
    d_finavalu[offset_1]  = dCaM;
    d_finavalu[offset_2]  = dCa2CaM;
    d_finavalu[offset_3]  = dCa4CaM;
    d_finavalu[offset_4]  = dCaMB;
    d_finavalu[offset_5]  = dCa2CaMB;
    d_finavalu[offset_6]  = dCa4CaMB;
    d_finavalu[offset_7]  = dPb2;
    d_finavalu[offset_8]  = dPb;
    d_finavalu[offset_9]  = dPt;
    d_finavalu[offset_10] = dPt2;
    d_finavalu[offset_11] = dPa;
    d_finavalu[offset_12] = dCa4CaN;
    d_finavalu[offset_13] = dCaMCa4CaN;
    d_finavalu[offset_14] = dCa2CaMCa4CaN;
    d_finavalu[offset_15] = dCa4CaMCa4CaN;

    // write to global variables for adjusting Ca buffering in EC coupling model
    d_finavalu[com_offset] =
        one_thousandth *
        ( (fp)2.0 * CaMKIItot * (rcnCKtt2 - rcnCKb2b) -
          (fp)2.0 * (rcn02 + rcn24 + rcn02B + rcn24B +
                     rcnCa4CaN + rcn02CaN + rcn24CaN) ); // [uM/msec]
    // d_finavalu[JCa] = 1; // [uM/msec]
}
