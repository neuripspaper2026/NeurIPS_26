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
    fp k20;  // [s^-1]
    fp k02;  // [uM^-2 s^-1]
    fp k42;  // [s^-1]
    fp k24;  // [uM^-2 s^-1]

    // CaM buffering (B) parameters
    fp k0Boff; // [s^-1]
    fp k0Bon;  // [uM^-1 s^-1] kon = koff/Kd
    fp k2Boff; // [s^-1]
    fp k2Bon;  // [uM^-1 s^-1]
    fp k4Boff; // [s^-1]
    fp k4Bon;  // [uM^-1 s^-1]

    // using thermodynamic constraints
    fp k20B; // [s^-1] thermo constraint on loop 1
    fp k02B; // [uM^-2 s^-1]
    fp k42B; // [s^-1] thermo constraint on loop 2
    fp k24B; // [uM^-2 s^-1]

    // Wi Wa Wt Wp
    fp kbi;   // [s^-1] (Ca4CaM dissocation from Wb)
    fp kib;   // [uM^-1 s^-1]
    fp kpp1;  // [s^-1] (PP1-dep dephosphorylation rates)
    fp Kmpp1; // [uM]
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
    fp kcanCaoff;   // [s^-1]
    fp kcanCaon;    // [uM^-1 s^-1]
    fp kcanCaM4on;  // [uM^-1 s^-1]
    fp kcanCaM4off; // [s^-1]
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

    //=====================================================================
    //	EXECUTION
    //=====================================================================

    // inputs
    Btot = d_params[params_offset + 1];
    CaMKIItot = d_params[params_offset + 2];
    CaNtot = d_params[params_offset + 3];
    PP1tot = d_params[params_offset + 4];
    K = d_params[16];
    Mg = d_params[17];

    // variable references
    offset_1 = valu_offset;
    offset_2 = valu_offset + 1;
    offset_3 = valu_offset + 2;
    offset_4 = valu_offset + 3;
    offset_5 = valu_offset + 4;
    offset_6 = valu_offset + 5;
    offset_7 = valu_offset + 6;
    offset_8 = valu_offset + 7;
    offset_9 = valu_offset + 8;
    offset_10 = valu_offset + 9;
    offset_11 = valu_offset + 10;
    offset_12 = valu_offset + 11;
    offset_13 = valu_offset + 12;
    offset_14 = valu_offset + 13;
    offset_15 = valu_offset + 14;

    // decoding input array
    CaM = d_initvalu[offset_1];
    Ca2CaM = d_initvalu[offset_2];
    Ca4CaM = d_initvalu[offset_3];
    CaMB = d_initvalu[offset_4];
    Ca2CaMB = d_initvalu[offset_5];
    Ca4CaMB = d_initvalu[offset_6];
    Pb2 = d_initvalu[offset_7];
    Pb = d_initvalu[offset_8];
    Pt = d_initvalu[offset_9];
    Pt2 = d_initvalu[offset_10];
    Pa = d_initvalu[offset_11];
    Ca4CaN = d_initvalu[offset_12];
    CaMCa4CaN = d_initvalu[offset_13];
    Ca2CaMCa4CaN = d_initvalu[offset_14];
    Ca4CaMCa4CaN = d_initvalu[offset_15];

    // Ca/CaM parameters
    if (Mg <= 1) {
        Kd02 = 0.0025f * (1.0f + K / 0.94f - Mg / 0.012f) *
               (1.0f + K / 8.1f + Mg / 0.022f); // [uM^2]
        Kd24 = 0.128f * (1.0f + K / 0.64f + Mg / 0.0014f) *
               (1.0f + K / 13.0f - Mg / 0.153f); // [uM^2]
    } else {
        Kd02 = 0.0025f * (1.0f + K / 0.94f - 1.0f / 0.012f + (Mg - 1.0f) / 0.060f) *
               (1.0f + K / 8.1f +
