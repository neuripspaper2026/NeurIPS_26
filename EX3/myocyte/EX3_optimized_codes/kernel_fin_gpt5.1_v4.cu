__device__ __forceinline__ void kernel_fin(fp * __restrict__ initvalu,
                                           int initvalu_offset_ecc,
                                           int initvalu_offset_Dyad,
                                           int initvalu_offset_SL,
                                           int initvalu_offset_Cyt,
                                           fp * __restrict__ parameter,
                                           fp * __restrict__ finavalu,
                                           fp JCaDyad,
                                           fp JCaSL,
                                           fp JCaCyt) {

    // decoded input parameters
    const fp BtotDyad      = parameter[2];
    const fp CaMKIItotDyad = parameter[3];

    // set variables (constants hoisted as const for better compiler optimization)
    const fp Vmyo  = (fp)2.1454e-11;  // [L]
    const fp Vdyad = (fp)1.7790e-14;  // [L]
    const fp VSL   = (fp)6.6013e-13;  // [L]
    const fp kSLmyo = (fp)8.587e-15;  // [L/msec]
    const fp k0Boff = (fp)0.0014;     // [s^-1]
    const fp k0Bon  = k0Boff / (fp)0.2;      // [uM^-1 s^-1] kon = koff/Kd
    const fp k2Boff = k0Boff / (fp)100.0;    // [s^-1]
    const fp k2Bon  = k0Bon;                 // [uM^-1 s^-1]
    const fp k4Bon  = k0Bon;                 // [uM^-1 s^-1]
    const fp one_thousandth = (fp)1e-3;

    // ADJUST ECC incorporate Ca buffering from CaM, convert JCaCyt from uM/msec to mM/msec
    const int ecc_off_35 = initvalu_offset_ecc + 35;
    const int ecc_off_36 = initvalu_offset_ecc + 36;
    const int ecc_off_37 = initvalu_offset_ecc + 37;

    finavalu[ecc_off_35] = finavalu[ecc_off_35] + one_thousandth * JCaDyad;
    finavalu[ecc_off_36] = finavalu[ecc_off_36] + one_thousandth * JCaSL;
    finavalu[ecc_off_37] = finavalu[ecc_off_37] + one_thousandth * JCaCyt;

    // Precompute frequently used base offsets
    const int dyad0 = initvalu_offset_Dyad;
    const int sl0   = initvalu_offset_SL;
    const int cyt0  = initvalu_offset_Cyt;

    // Load Dyad CaM-related states into registers to reduce global memory traffic
    const fp iDyad0  = initvalu[dyad0 + 0];
    const fp iDyad1  = initvalu[dyad0 + 1];
    const fp iDyad2  = initvalu[dyad0 + 2];
    const fp iDyad3  = initvalu[dyad0 + 3];
    const fp iDyad4  = initvalu[dyad0 + 4];
    const fp iDyad5  = initvalu[dyad0 + 5];
    const fp iDyad6  = initvalu[dyad0 + 6];
    const fp iDyad7  = initvalu[dyad0 + 7];
    const fp iDyad8  = initvalu[dyad0 + 8];
    const fp iDyad9  = initvalu[dyad0 + 9];
    const fp iDyad12 = initvalu[dyad0 + 12];
    const fp iDyad13 = initvalu[dyad0 + 13];
    const fp iDyad14 = initvalu[dyad0 + 14];

    // CaMtotDyad accumulation using fused-add style for better ILP
    fp CaMtotDyad = iDyad0 + iDyad1 + iDyad2 + iDyad3 + iDyad4 + iDyad5;
    fp CaMKII_sum = iDyad6 + iDyad7 + iDyad8 + iDyad9;
    CaMtotDyad += CaMKIItotDyad * CaMKII_sum;
    CaMtotDyad += iDyad12 + iDyad13 + iDyad14;

    const fp Bdyad = BtotDyad - CaMtotDyad; // [uM dyad]

    // Load SL states used in diffusion terms
    const fp iSL0 = initvalu[sl0 + 0];
    const fp iSL1 = initvalu[sl0 + 1];
    const fp iSL2 = initvalu[sl0 + 2];

    // Dyad fluxes (scaled by 1e-3)
    const fp J_cam_dyadSL =
        one_thousandth * (k0Boff * iDyad0 - k0Bon * Bdyad * iSL0); // [uM/msec dyad]
    const fp J_ca2cam_dyadSL =
        one_thousandth * (k2Boff * iDyad1 - k2Bon * Bdyad * iSL1); // [uM/msec dyad]
    const fp J_ca4cam_dyadSL =
        one_thousandth * (k2Boff * iDyad2 - k4Bon * Bdyad * iSL2); // [uM/msec dyad]

    // Load Cyt states used in SL-myo fluxes
    const fp iCyt0 = initvalu[cyt0 + 0];
    const fp iCyt1 = initvalu[cyt0 + 1];
    const fp iCyt2 = initvalu[cyt0 + 2];

    // SL-myo fluxes [umol/msec]
    const fp dSL0 = iSL0 - iCyt0;
    const fp dSL1 = iSL1 - iCyt1;
    const fp dSL2 = iSL2 - iCyt2;

    const fp J_cam_SLmyo   = kSLmyo * dSL0;
    const fp J_ca2cam_SLmyo= kSLmyo * dSL1;
    const fp J_ca4cam_SLmyo= kSLmyo * dSL2;

    // ADJUST CAM Dyad
    finavalu[dyad0 + 0] = finavalu[dyad0 + 0] - J_cam_dyadSL;
    finavalu[dyad0 + 1] = finavalu[dyad0 + 1] - J_ca2cam_dyadSL;
    finavalu[dyad0 + 2] = finavalu[dyad0 + 2] - J_ca4cam_dyadSL;

    // Precompute geometric ratios for SL updates
    const fp Vdyad_over_VSL = Vdyad / VSL;
    const fp inv_VSL        = (fp)1.0 / VSL;

    // ADJUST CAM SL
    finavalu[sl0 + 0] = finavalu[sl0 + 0] +
                        J_cam_dyadSL * Vdyad_over_VSL -
                        J_cam_SLmyo * inv_VSL;
    finavalu[sl0 + 1] = finavalu[sl0 + 1] +
                        J_ca2cam_dyadSL * Vdyad_over_VSL -
                        J_ca2cam_SLmyo * inv_VSL;
    finavalu[sl0 + 2] = finavalu[sl0 + 2] +
                        J_ca4cam_dyadSL * Vdyad_over_VSL -
                        J_ca4cam_SLmyo * inv_VSL;

    // ADJUST CAM Cyt
    const fp inv_Vmyo = (fp)1.0 / Vmyo;
    finavalu[cyt0 + 0] = finavalu[cyt0 + 0] + J_cam_SLmyo * inv_Vmyo;
    finavalu[cyt0 + 1] = finavalu[cyt0 + 1] + J_ca2cam_SLmyo * inv_Vmyo;
    finavalu[cyt0 + 2] = finavalu[cyt0 + 2] + J_ca4cam_SLmyo * inv_Vmyo;
}
