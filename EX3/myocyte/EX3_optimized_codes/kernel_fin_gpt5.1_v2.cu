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

    //=====================================================================
    //	VARIABLES
    //=====================================================================

    // decoded input parameters
    const fp BtotDyad      = parameter[2];
    const fp CaMKIItotDyad = parameter[3];

    // compute variables (constants hoisted as const for potential FMA use)
    const fp Vmyo  = (fp)2.1454e-11;  // [L]
    const fp Vdyad = (fp)1.7790e-14;  // [L]
    const fp VSL   = (fp)6.6013e-13;  // [L]

    const fp kSLmyo = (fp)8.587e-15;  // [L/msec]
    const fp k0Boff = (fp)0.0014;     // [s^-1]
    const fp k0Bon  = k0Boff / (fp)0.2;   // [uM^-1 s^-1] kon = koff/Kd
    const fp k2Boff = k0Boff / (fp)100.0; // [s^-1]
    const fp k2Bon  = k0Bon;              // [uM^-1 s^-1]
    const fp k4Bon  = k0Bon;              // [uM^-1 s^-1]

    const fp inv_1000 = (fp)1e-3;

    //=====================================================================
    //	COMPUTATION
    //=====================================================================

    // ADJUST ECC incorporate Ca buffering from CaM, convert JCaCyt from uM/msec to mM/msec
    const int ecc35 = initvalu_offset_ecc + 35;
    const int ecc36 = ecc35 + 1;
    const int ecc37 = ecc35 + 2;

    finavalu[ecc35] += inv_1000 * JCaDyad;
    finavalu[ecc36] += inv_1000 * JCaSL;
    finavalu[ecc37] += inv_1000 * JCaCyt;

    // precompute commonly used base pointers
    fp * __restrict__ initDyad = initvalu + initvalu_offset_Dyad;
    fp * __restrict__ initSL   = initvalu + initvalu_offset_SL;
    fp * __restrict__ initCyt  = initvalu + initvalu_offset_Cyt;

    fp * __restrict__ finDyad = finavalu + initvalu_offset_Dyad;
    fp * __restrict__ finSL   = finavalu + initvalu_offset_SL;
    fp * __restrict__ finCyt  = finavalu + initvalu_offset_Cyt;

    // incorporate CaM diffusion between compartments
    const fp CaMKIItotDyad_loc = CaMKIItotDyad;

    const fp CaM_dyad      = initDyad[0];
    const fp Ca2CaM_dyad   = initDyad[1];
    const fp Ca4CaM_dyad   = initDyad[2];
    const fp CaMB_dyad     = initDyad[3];
    const fp Ca2CaMB_dyad  = initDyad[4];
    const fp Ca4CaMB_dyad  = initDyad[5];
    const fp CK0_dyad      = initDyad[6];
    const fp CK1_dyad      = initDyad[7];
    const fp CK2_dyad      = initDyad[8];
    const fp CK3_dyad      = initDyad[9];
    const fp CaMCK0_dyad   = initDyad[12];
    const fp CaMCK1_dyad   = initDyad[13];
    const fp CaMCK2_dyad   = initDyad[14];

    const fp CaMKIIsum = CaMKIItotDyad_loc *
                         (CK0_dyad + CK1_dyad + CK2_dyad + CK3_dyad);

    const fp CaMtotDyad =
        CaM_dyad + Ca2CaM_dyad + Ca4CaM_dyad +
        CaMB_dyad + Ca2CaMB_dyad + Ca4CaMB_dyad +
        CaMKIIsum +
        CaMCK0_dyad + CaMCK1_dyad + CaMCK2_dyad;

    const fp Bdyad = BtotDyad - CaMtotDyad; // [uM dyad]

    // CAM fluxes between Dyad and SL (scaled by 1e-3)
    const fp CaM_SL     = initSL[0];
    const fp Ca2CaM_SL  = initSL[1];
    const fp Ca4CaM_SL  = initSL[2];

    const fp J_cam_dyadSL =
        inv_1000 * (k0Boff * CaM_dyad - k0Bon * Bdyad * CaM_SL);          // [uM/msec dyad]
    const fp J_ca2cam_dyadSL =
        inv_1000 * (k2Boff * Ca2CaM_dyad - k2Bon * Bdyad * Ca2CaM_SL);    // [uM/msec dyad]
    const fp J_ca4cam_dyadSL =
        inv_1000 * (k2Boff * Ca4CaM_dyad - k4Bon * Bdyad * Ca4CaM_SL);    // [uM/msec dyad]

    // CAM fluxes between SL and myoplasm (umol/msec)
    const fp CaM_Cyt    = initCyt[0];
    const fp Ca2CaM_Cyt = initCyt[1];
    const fp Ca4CaM_Cyt = initCyt[2];

    const fp J_cam_SLmyo =
        kSLmyo * (CaM_SL    - CaM_Cyt);
    const fp J_ca2cam_SLmyo =
        kSLmyo * (Ca2CaM_SL - Ca2CaM_Cyt);
    const fp J_ca4cam_SLmyo =
        kSLmyo * (Ca4CaM_SL - Ca4CaM_Cyt);

    // ADJUST CAM Dyad
    finDyad[0] -= J_cam_dyadSL;
    finDyad[1] -= J_ca2cam_dyadSL;
    finDyad[2] -= J_ca4cam_dyadSL;

    // ADJUST CAM SL
    const fp Vdyad_over_VSL = Vdyad / VSL;
    const fp inv_VSL        = (fp)1.0 / VSL;

    finSL[0] += J_cam_dyadSL   * Vdyad_over_VSL - J_cam_SLmyo   * inv_VSL;
    finSL[1] += J_ca2cam_dyadSL * Vdyad_over_VSL - J_ca2cam_SLmyo * inv_VSL;
    finSL[2] += J_ca4cam_dyadSL * Vdyad_over_VSL - J_ca4cam_SLmyo * inv_VSL;

    // ADJUST CAM Cyt
    const fp inv_Vmyo = (fp)1.0 / Vmyo;

    finCyt[0] += J_cam_SLmyo   * inv_Vmyo;
    finCyt[1] += J_ca2cam_SLmyo * inv_Vmyo;
    finCyt[2] += J_ca4cam_SLmyo * inv_Vmyo;
}
