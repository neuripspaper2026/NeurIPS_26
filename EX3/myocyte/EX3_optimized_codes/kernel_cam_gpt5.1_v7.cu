__device__ void kernel_cam(fp timeinst, fp * __restrict__ d_initvalu, fp * __restrict__ d_finavalu,
                           int valu_offset, fp * __restrict__ d_params, int params_offset,
                           fp * __restrict__ d_com, int com_offset, fp Ca) {

    //=====================================================================
    //	VARIABLES
    //=====================================================================

    // inputs
    // fp CaMtot;
    fp Btot;
    fp CaMKIItot;
    fp CaNtot;
    fp PP1tot;
    fp K;
    fp Mg;

    // variable references
    int offset_1;
    int offset_2;
    int offset_3;
    int offset_4;
    int offset_5;
    int offset_6;
    int offset_7;
    int offset_8;
    int offset_9;
    int offset_10;
    int offset_11;
    int offset_12;
    int offset_13;
    int offset_14;
    int offset_15;

    // decoding input array
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
    fp Kd02; // [uM^2]
    fp Kd24; // [uM^2]
    const fp k20 = (fp)10.0;   // [s^-1]
    fp k02;  // [uM^-2 s^-1]
    const fp k42 = (fp)500.0;  // [s^-1]
    fp k24;  // [uM^-2 s^-1]

    // CaM buffering (B) parameters
    const fp k0Boff = (fp)0.0014;       // [s^-1]
    const fp k0Bon  = k0Boff / (fp)0.2; // [uM^-1 s^-1] kon = koff/Kd
    const fp k2Boff = k0Boff / (fp)100.0; // [s^-1]
    const fp k2Bon  = k0Bon;            // [uM^-1 s^-1]
    const fp k4Boff = k2Boff;           // [s^-1]
    const fp k4Bon  = k0Bon;            // [uM^-1 s^-1]

    // using thermodynamic constraints
    fp k20B; // [s^-1] thermo constraint on loop 1
    fp k02B; // [uM^-2 s^-1]
    fp k42B; // [s^-1] thermo constraint on loop 2
    fp k24B; // [uM^-2 s^-1]

    // Wi Wa Wt Wp
    const fp kbi   = (fp)2.2;                  // [s^-1] (Ca4CaM dissocation from Wb)
    const fp kib   = kbi / (fp)33.5e-3;        // [uM^-1 s^-1]
    const fp kpp1  = (fp)1.72;                 // [s^-1] (PP1-dep dephosphorylation rates)
    const fp Kmpp1 = (fp)11.5;                 // [uM]
    fp kib2;
    fp kb2i;
    fp kb24;
    fp kb42;
    fp kta; // [s^-1] (Ca4CaM dissociation from Wt)
    fp kat; // [uM^-1 s^-1] (Ca4CaM reassociation with Wa)
    fp kt42;
    fp kt24;
    fp kat2;
    fp kt2a;

    // CaN parameters
    const fp kcanCaoff = (fp)1.0;                // [s^-1]
    const fp kcanCaon  = kcanCaoff / (fp)0.5;    // [uM^-1 s^-1]
    const fp kcanCaM4on  = (fp)46.0;             // [uM^-1 s^-1]
    const fp kcanCaM4off = (fp)0.0013;           // [s^-1]
    const fp kcanCaM2on  = kcanCaM4on;
    const fp kcanCaM2off = (fp)2508.0 * kcanCaM4off;
    const fp kcanCaM0on  = kcanCaM4on;
    const fp kcanCaM0off = (fp)165.0 * kcanCaM2off;
    fp k02can;
    fp k20can;
    fp k24can;
    fp k42can;

    // precomputed Ca powers
    const fp Ca2 = Ca * Ca;

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
    fp dPb2; // Pb2
    fp dPb;  // Pb
    fp dPt;  // Pt
    fp dPt2; // Pt2
    fp dPa;  // Pa

    // CaN equations
    fp dCa4CaN;       // Ca4CaN
    fp dCaMCa4CaN;    // CaMCa4CaN
    fp dCa2CaMCa4CaN; // Ca2CaMCa4CaN
    fp dCa4CaMCa4CaN; // Ca4CaMCa4CaN

    const fp scale = (fp)1e-3;

    //=====================================================================
    //	EXECUTION
    //=====================================================================

    // inputs
    // CaMtot = d_params[params_offset];
    Btot      = d_params[params_offset + 1];
    CaMKIItot = d_params[params_offset + 2];
    CaNtot    = d_params[params_offset + 3];
    PP1tot    = d_params[params_offset + 4];
    K         = d_params[16];
    Mg        = d_params[17];

    // variable references
    offset_1  = valu_offset;
    offset_2  = valu_offset + 1;
    offset_3  = valu_offset + 2;
    offset_4  = valu_offset + 3;
    offset_5  = valu_offset + 4;
    offset_6  = valu_offset + 5;
    offset_7  = valu_offset + 6;
    offset_8  = valu_offset + 7;
    offset_9  = valu_offset + 8;
    offset_10 = valu_offset + 9;
    offset_11 = valu_offset + 10;
    offset_12 = valu_offset + 11;
    offset_13 = valu_offset + 12;
    offset_14 = valu_offset + 13;
    offset_15 = valu_offset + 14;

    // decoding input array
    CaM            = d_initvalu[offset_1];
    Ca2CaM         = d_initvalu[offset_2];
    Ca4CaM         = d_initvalu[offset_3];
    CaMB           = d_initvalu[offset_4];
    Ca2CaMB        = d_initvalu[offset_5];
    Ca4CaMB        = d_initvalu[offset_6];
    Pb2            = d_initvalu[offset_7];
    Pb             = d_initvalu[offset_8];
    Pt             = d_initvalu[offset_9];
    Pt2            = d_initvalu[offset_10];
    Pa             = d_initvalu[offset_11];
    Ca4CaN         = d_initvalu[offset_12];
    CaMCa4CaN      = d_initvalu[offset_13];
    Ca2CaMCa4CaN   = d_initvalu[offset_14];
    Ca4CaMCa4CaN   = d_initvalu[offset_15];

    // Ca/CaM parameters
    if (Mg <= (fp)1.0) {
        Kd02 = (fp)0.0025 * ( (fp)1.0 + K / (fp)0.94 - Mg / (fp)0.012 ) *
                          ( (fp)1.0 + K / (fp)8.1 + Mg / (fp)0.022 ); // [uM^2]
        Kd24 = (fp)0.128 * ( (fp)1.0 + K / (fp)0.64 + Mg / (fp)0.0014 ) *
                          ( (fp)1.0 + K / (fp)13.0 - Mg / (fp)0.153 ); // [uM^2]
    } else {
        const fp Mg_m1 = Mg - (fp)1.0;
        Kd02 = (fp)0.0025 *
               ( (fp)1.0 + K / (fp)0.94 - (fp)1.0 / (fp)0.012 + Mg_m1 / (fp)0.060 ) *
               ( (fp)1.0 + K / (fp)8.1 + (fp)1.0 / (fp)0.022 + Mg_m1 / (fp)0.068 ); // [uM^2]
        Kd24 = (fp)0.128 *
               ( (fp)1.0 + K / (fp)0.64 + (fp)1.0 / (fp)0.0014 + Mg_m1 / (fp)0.005 ) *
               ( (fp)1.0 + K / (fp)13.0 - (fp)1.0 / (fp)0.153 + Mg_m1 / (fp)0.150 ); // [uM^2]
    }

    k02 = k20 / Kd02; // [uM^-2 s^-1]
    k24 = k42 / Kd24; // [uM^-2 s^-1]

    // CaM buffering (B) parameters already set as consts above

    // using thermodynamic constraints
    k20B = k20 / (fp)100.0; // [s^-1] thermo constraint on loop 1
    k02B = k02;             // [uM^-2 s^-1]
    k42B = k42;             // [s^-1] thermo constraint on loop 2
    k24B = k24;             // [uM^-2 s^-1]

    // Wi Wa Wt Wp
    kib2 = kib;
    kb2i = kib2 * (fp)5.0;
    kb24 = k24;
    kb42 = k42 * (fp)33.5e-3 / (fp)5.0;
    kta  = kbi / (fp)1000.0;   // [s^-1] (Ca4CaM dissociation from Wt)
    kat  = kib;                // [uM^-1 s^-1] (Ca4CaM reassociation with Wa)
    kt42 = k42 * (fp)33.5e-6 / (fp)5.0;
    kt24 = k24;
    kat2 = kib;
    kt2a = kib * (fp)5.0;

    // CaN parameters
    k02can = k02;
    k20can = k20 / (fp)165.0;
    k24can = k24;
    k42can = k20 / (fp)2508.0;

    // CaM Reaction fluxes
    rcn02 = k02 * Ca2 * CaM   - k20 * Ca2CaM;
    rcn24 = k24 * Ca2 * Ca2CaM - k42 * Ca4CaM;

    // CaM buffer fluxes
    B      = Btot - CaMB - Ca2CaMB - Ca4CaMB;
    rcn02B = k02B * Ca2 * CaMB    - k20B * Ca2CaMB;
    rcn24B = k24B * Ca2 * Ca2CaMB - k42B * Ca4CaMB;
    rcn0B  = k0Bon * CaM    * B   - k0Boff * CaMB;
    rcn2B  = k2Bon * Ca2CaM * B   - k2Boff * Ca2CaMB;
    rcn4B  = k4Bon * Ca4CaM * B   - k4Boff * Ca4CaMB;

    // CaN reaction fluxes
    Ca2CaN    = CaNtot - Ca4CaN - CaMCa4CaN - Ca2CaMCa4CaN - Ca4CaMCa4CaN;
    rcnCa4CaN = kcanCaon  * Ca2 * Ca2CaN      - kcanCaoff * Ca4CaN;
    rcn02CaN  = k02can    * Ca2 * CaMCa4CaN   - k20can * Ca2CaMCa4CaN;
    rcn24CaN  = k24can    * Ca2 * Ca2CaMCa4CaN - k42can * Ca4CaMCa4CaN;
    rcn0CaN   = kcanCaM0on * CaM    * Ca4CaN  - kcanCaM0off * CaMCa4CaN;
    rcn2CaN   = kcanCaM2on * Ca2CaM * Ca4CaN  - kcanCaM2off * Ca2CaMCa4CaN;
    rcn4CaN   = kcanCaM4on * Ca4CaM * Ca4CaN  - kcanCaM4off * Ca4CaMCa4CaN;

    // CaMKII reaction fluxes
    Pix       = (fp)1.0 - Pb2 - Pb - Pt - Pt2 - Pa;
    rcnCKib2  = kib2 * Ca2CaM * Pix - kb2i * Pb2;
    rcnCKb2b  = kb24 * Ca2 * Pb2    - kb42 * Pb;
    rcnCKib   = kib  * Ca4CaM * Pix - kbi  * Pb;
    T         = Pb + Pt + Pt2 + Pa;
    {
        const fp T2 = T * T;
        const fp T3 = T2 * T;
        kbt = (fp)0.055 * T + (fp)0.0074 * T2 + (fp)0.015 * T3;
    }
    rcnCKbt   = kbt * Pb - kpp1 * PP1tot * Pt  / (Kmpp1 + CaMKIItot * Pt);
    rcnCKtt2  = kt42 * Pt  - kt24 * Ca2 * Pt2;
    rcnCKta   = kta  * Pt  - kat  * Ca4CaM * Pa;
    rcnCKt2a  = kt2a * Pt2 - kat2 * Ca2CaM * Pa;
    rcnCKt2b2 = kpp1 * PP1tot * Pt2 / (Kmpp1 + CaMKIItot * Pt2);
    rcnCKai   = kpp1 * PP1tot * Pa  / (Kmpp1 + CaMKIItot * Pa);

    // CaM equations
    dCaM      = scale * ( -rcn02 - rcn0B - rcn0CaN );
    dCa2CaM   = scale * ( rcn02 - rcn24 - rcn2B - rcn2CaN +
                          CaMKIItot * ( -rcnCKib2 + rcnCKt2a ) );
    dCa4CaM   = scale * ( rcn24 - rcn4B - rcn4CaN +
                          CaMKIItot * ( -rcnCKib + rcnCKta ) );
    dCaMB     = scale * ( rcn0B - rcn02B );
    dCa2CaMB  = scale * ( rcn02B + rcn2B - rcn24B );
    dCa4CaMB  = scale * ( rcn24B + rcn4B );

    // CaMKII equations
    dPb2      = scale * ( rcnCKib2 - rcnCKb2b + rcnCKt2b2 ); // Pb2
    dPb       = scale * ( rcnCKib + rcnCKb2b - rcnCKbt );    // Pb
    dPt       = scale * ( rcnCKbt - rcnCKta - rcnCKtt2 );    // Pt
    dPt2      = scale * ( rcnCKtt2 - rcnCKt2a - rcnCKt2b2 ); // Pt2
    dPa       = scale * ( rcnCKta + rcnCKt2a - rcnCKai );    // Pa

    // CaN equations
    dCa4CaN        = scale * ( rcnCa4CaN - rcn0CaN - rcn2CaN - rcn4CaN ); // Ca4CaN
    dCaMCa4CaN     = scale * ( rcn0CaN - rcn02CaN );                     // CaMCa4CaN
    dCa2CaMCa4CaN  = scale * ( rcn2CaN + rcn02CaN - rcn24CaN );          // Ca2CaMCa4CaN
    dCa4CaMCa4CaN  = scale * ( rcn4CaN + rcn24CaN );                     // Ca4CaMCa4CaN

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
        scale * ( (fp)2.0 * CaMKIItot * ( rcnCKtt2 - rcnCKb2b ) -
                  (fp)2.0 * ( rcn02 + rcn24 + rcn02B + rcn24B +
                              rcnCa4CaN + rcn02CaN + rcn24CaN ) ); // [uM/msec]
    // d_finavalu[JCa] = 1; // [uM/msec]
}
