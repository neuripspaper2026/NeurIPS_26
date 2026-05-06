__device__ void kernel_fin(const fp * __restrict__ initvalu, int initvalu_offset_ecc, int initvalu_offset_Dyad,
                           int initvalu_offset_SL, int initvalu_offset_Cyt, const fp * __restrict__ parameter,
                           fp * __restrict__ finavalu, fp JCaDyad, fp JCaSL, fp JCaCyt) {

    //=====================================================================
    //	VARIABLES
    //=====================================================================

    // decoded input parameters
    fp BtotDyad;      //
    fp CaMKIItotDyad; //

    // compute variables
    fp Vmyo;  // [L]
    fp Vdyad; // [L]
    fp VSL;   // [L]
    // fp kDyadSL;																			//
    // [L/msec]
    fp kSLmyo; // [L/msec]
    fp k0Boff; // [s^-1]
    fp k0Bon;  // [uM^-1 s^-1] kon = koff/Kd
    fp k2Boff; // [s^-1]
    fp k2Bon;  // [uM^-1 s^-1]
    // fp k4Boff;																			//
    // [s^-1]
    fp k4Bon; // [uM^-1 s^-1]
    fp CaMtotDyad;
    fp Bdyad;           // [uM dyad]
    fp J_cam_dyadSL;    // [uM/msec dyad]
    fp J_ca2cam_dyadSL; // [uM/msec dyad]
    fp J_ca4cam_dyadSL; // [uM/msec dyad]
    fp J_cam_SLmyo;     // [umol/msec]
    fp J_ca2cam_SLmyo;  // [umol/msec]
    fp J_ca4cam_SLmyo;  // [umol/msec]

    //=====================================================================
    //	COMPUTATION
    //=====================================================================

    // decoded input parameters
    BtotDyad      = parameter[2]; //
    CaMKIItotDyad = parameter[3]; //

    // set variables
    Vmyo  = (fp)2.1454e-11;  // [L]
    Vdyad = (fp)1.7790e-14;  // [L]
    VSL   = (fp)6.6013e-13;  // [L]
    // kDyadSL = 3.6363e-16;																//
    // [L/msec]
    kSLmyo = (fp)8.587e-15;     // [L/msec]
    k0Boff = (fp)0.0014;        // [s^-1]
    k0Bon  = k0Boff / (fp)0.2;  // [uM^-1 s^-1] kon = koff/Kd
    k2Boff = k0Boff / (fp)100;  // [s^-1]
    k2Bon  = k0Bon;             // [uM^-1 s^-1]
    // k4Boff = k2Boff;																	//
    // [s^-1]
    k4Bon = k0Bon; // [uM^-1 s^-1]

    // ADJUST ECC incorporate Ca buffering from CaM, convert JCaCyt from uM/msec
    // to mM/msec
    const fp one_over_1000 = (fp)1e-3;

    const int ecc35 = initvalu_offset_ecc + 35;
    const int ecc36 = initvalu_offset_ecc + 36;
    const int ecc37 = initvalu_offset_ecc + 37;

    finavalu[ecc35] = finavalu[ecc35] + one_over_1000 * JCaDyad;
    finavalu[ecc36] = finavalu[ecc36] + one_over_1000 * JCaSL;
    finavalu[ecc37] = finavalu[ecc37] + one_over_1000 * JCaCyt;

    // precompute commonly used offsets
    const int offDyad0 = initvalu_offset_Dyad;
    const int offSL0   = initvalu_offset_SL;
    const int offCyt0  = initvalu_offset_Cyt;

    // load Dyad states into registers
    const fp Dyad0  = initvalu[offDyad0 + 0];
    const fp Dyad1  = initvalu[offDyad0 + 1];
    const fp Dyad2  = initvalu[offDyad0 + 2];
    const fp Dyad3  = initvalu[offDyad0 + 3];
    const fp Dyad4  = initvalu[offDyad0 + 4];
    const fp Dyad5  = initvalu[offDyad0 + 5];
    const fp Dyad6  = initvalu[offDyad0 + 6];
    const fp Dyad7  = initvalu[offDyad0 + 7];
    const fp Dyad8  = initvalu[offDyad0 + 8];
    const fp Dyad9  = initvalu[offDyad0 + 9];
    const fp Dyad12 = initvalu[offDyad0 + 12];
    const fp Dyad13 = initvalu[offDyad0 + 13];
    const fp Dyad14 = initvalu[offDyad0 + 14];

    // incorporate CaM diffusion between compartments
    CaMtotDyad =
        Dyad0 + Dyad1 + Dyad2 + Dyad3 + Dyad4 + Dyad5 +
        CaMKIItotDyad * (Dyad6 + Dyad7 + Dyad8 + Dyad9) +
        Dyad12 + Dyad13 + Dyad14;

    Bdyad = BtotDyad - CaMtotDyad; // [uM dyad]

    // load SL and Cyt states into registers for first three species
    const fp SL0  = initvalu[offSL0 + 0];
    const fp SL1  = initvalu[offSL0 + 1];
    const fp SL2  = initvalu[offSL0 + 2];
    const fp Cyt0 = initvalu[offCyt0 + 0];
    const fp Cyt1 = initvalu[offCyt0 + 1];
    const fp Cyt2 = initvalu[offCyt0 + 2];

    // dyad <-> SL fluxes (with fused multiplication by 1e-3)
    const fp Bdyad_SL0 = Bdyad * SL0;
    const fp Bdyad_SL1 = Bdyad * SL1;
    const fp Bdyad_SL2 = Bdyad * SL2;

    J_cam_dyadSL =
        one_over_1000 *
        (k0Boff * Dyad0 - k0Bon * Bdyad_SL0); // [uM/msec dyad]
    J_ca2cam_dyadSL =
        one_over_1000 *
        (k2Boff * Dyad1 - k2Bon * Bdyad_SL1); // [uM/msec dyad]
    J_ca4cam_dyadSL =
        one_over_1000 *
        (k2Boff * Dyad2 - k4Bon * Bdyad_SL2); // [uM/msec dyad]

    // SL <-> myo fluxes
    const fp SL0_minus_Cyt0 = SL0 - Cyt0;
    const fp SL1_minus_Cyt1 = SL1 - Cyt1;
    const fp SL2_minus_Cyt2 = SL2 - Cyt2;

    J_cam_SLmyo    = kSLmyo * SL0_minus_Cyt0; // [umol/msec]
    J_ca2cam_SLmyo = kSLmyo * SL1_minus_Cyt1; // [umol/msec]
    J_ca4cam_SLmyo = kSLmyo * SL2_minus_Cyt2; // [umol/msec]

    // ADJUST CAM Dyad
    finavalu[offDyad0 + 0] -= J_cam_dyadSL;
    finavalu[offDyad0 + 1] -= J_ca2cam_dyadSL;
    finavalu[offDyad0 + 2] -= J_ca4cam_dyadSL;

    // ADJUST CAM Sl
    const fp Vdyad_over_VSL = Vdyad / VSL;
    const fp invVSL         = (fp)1.0 / VSL;

    finavalu[offSL0 + 0] += J_cam_dyadSL * Vdyad_over_VSL - J_cam_SLmyo * invVSL;
    finavalu[offSL0 + 1] += J_ca2cam_dyadSL * Vdyad_over_VSL - J_ca2cam_SLmyo * invVSL;
    finavalu[offSL0 + 2] += J_ca4cam_dyadSL * Vdyad_over_VSL - J_ca4cam_SLmyo * invVSL;

    // ADJUST CAM Cyt
    const fp invVmyo = (fp)1.0 / Vmyo;

    finavalu[offCyt0 + 0] += J_cam_SLmyo * invVmyo;
    finavalu[offCyt0 + 1] += J_ca2cam_SLmyo * invVmyo;
    finavalu[offCyt0 + 2] += J_ca4cam_SLmyo * invVmyo;
}
