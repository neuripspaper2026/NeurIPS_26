__device__ void kernel_ecc(fp timeinst, fp * __restrict__ d_initvalu, fp * __restrict__ d_finavalu,
                           int valu_offset, fp * __restrict__ d_params) {

    //=====================================================================
    //  VARIABLES
    //=====================================================================

    // input parameters
    fp cycleLength;

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
    int offset_16;
    int offset_17;
    int offset_18;
    int offset_19;
    int offset_20;
    int offset_21;
    int offset_22;
    int offset_23;
    int offset_24;
    int offset_25;
    int offset_26;
    int offset_27;
    int offset_28;
    int offset_29;
    int offset_30;
    int offset_31;
    int offset_32;
    int offset_33;
    int offset_34;
    int offset_35;
    int offset_36;
    int offset_37;
    int offset_38;
    int offset_39;
    int offset_40;
    int offset_41;
    int offset_42;
    int offset_43;
    int offset_44;
    int offset_45;
    int offset_46;

    // stored input array
    fp d_initvalu_1;
    fp d_initvalu_2;
    fp d_initvalu_3;
    fp d_initvalu_4;
    fp d_initvalu_5;
    fp d_initvalu_6;
    fp d_initvalu_7;
    fp d_initvalu_8;
    fp d_initvalu_9;
    fp d_initvalu_10;
    fp d_initvalu_11;
    fp d_initvalu_12;
    fp d_initvalu_13;
    fp d_initvalu_14;
    fp d_initvalu_15;
    fp d_initvalu_16;
    fp d_initvalu_17;
    fp d_initvalu_18;
    fp d_initvalu_19;
    fp d_initvalu_20;
    fp d_initvalu_21;
    // fp d_initvalu_22;
    fp d_initvalu_23;
    fp d_initvalu_24;
    fp d_initvalu_25;
    fp d_initvalu_26;
    fp d_initvalu_27;
    fp d_initvalu_28;
    fp d_initvalu_29;
    fp d_initvalu_30;
    fp d_initvalu_31;
    fp d_initvalu_32;
    fp d_initvalu_33;
    fp d_initvalu_34;
    fp d_initvalu_35;
    fp d_initvalu_36;
    fp d_initvalu_37;
    fp d_initvalu_38;
    fp d_initvalu_39;
    fp d_initvalu_40;
    // fp d_initvalu_41;
    // fp d_initvalu_42;
    // fp d_initvalu_43;
    // fp d_initvalu_44;
    // fp d_initvalu_45;
    // fp d_initvalu_46;

    // matlab constants undefined in c
    fp pi;

    // Constants
    fp R;    // [J/kmol*K]
    fp Frdy; // [C/mol]
    fp Temp; // [K] 310
    fp FoRT; //
    fp Cmem; // [F] membrane capacitance
    fp Qpow;

    // Cell geometry
    fp cellLength; // cell length [um]
    fp cellRadius; // cell radius [um]
    fp Vcell;      // [L]
    fp Vmyo;
    fp Vsr;
    fp Vsl;
    fp Vjunc;
    fp J_ca_juncsl; // [L/msec]
    fp J_ca_slmyo;  // [L/msec]
    fp J_na_juncsl; // [L/msec]
    fp J_na_slmyo;  // [L/msec]

    // Fractional currents in compartments
    fp Fjunc;
    fp Fsl;
    fp Fjunc_CaL;
    fp Fsl_CaL;

    // Fixed ion concentrations
    fp Cli; // Intracellular Cl  [mM]
    fp Clo; // Extracellular Cl  [mM]
    fp Ko;  // Extracellular K   [mM]
    fp Nao; // Extracellular Na  [mM]
    fp Cao; // Extracellular Ca  [mM]
    fp Mgi; // Intracellular Mg  [mM]

    // Nernst Potentials
    fp ena_junc; // [mV]
    fp ena_sl;   // [mV]
    fp ek;       // [mV]
    fp eca_junc; // [mV]
    fp eca_sl;   // [mV]
    fp ecl;      // [mV]

    // Na transport parameters
    fp GNa;     // [mS/uF]
    fp GNaB;    // [mS/uF]
    fp IbarNaK; // [uA/uF]
    fp KmNaip;  // [mM]
    fp KmKo;    // [mM]

    // K current parameters
    fp pNaK;
    fp GtoSlow; // [mS/uF]
    fp GtoFast; // [mS/uF]
    fp gkp;

    // Cl current parameters
    fp GClCa;  // [mS/uF]
    fp GClB;   // [mS/uF]
    fp KdClCa; // [mM]

    // I_Ca parameters
    fp pNa;    // [cm/sec]
    fp pCa;    // [cm/sec]
    fp pK;     // [cm/sec]
    fp Q10CaL; // [none]

    // Ca transport parameters
    fp IbarNCX;   // [uA/uF]
    fp KmCai;     // [mM]
    fp KmCao;     // [mM]
    fp KmNai;     // [mM]
    fp KmNao;     // [mM]
    fp ksat;      // [none]
    fp nu;        // [none]
    fp Kdact;     // [mM]
    fp Q10NCX;    // [none]
    fp IbarSLCaP; // [uA/uF]
    fp KmPCa;     // [mM]
    fp GCaB;      // [uA/uF]
    fp Q10SLCaP;  // [none]

    // SR flux parameters
    fp Q10SRCaP;   // [none]
    fp Vmax_SRCaP; // [mM/msec]
    fp Kmf;        // [mM]
    fp Kmr;        // [mM]
    fp hillSRCaP;  // [mM]
    fp ks;         // [1/ms]
    fp koCa;       // [mM^-2 1/ms]
    fp kom;        // [1/ms]
    fp kiCa;       // [1/mM/ms]
    fp kim;        // [1/ms]
    fp ec50SR;     // [mM]

    // Buffering parameters
    fp Bmax_Naj;     // [mM]
    fp Bmax_Nasl;    // [mM]
    fp koff_na;      // [1/ms]
    fp kon_na;       // [1/mM/ms]
    fp Bmax_TnClow;  // [mM]
    fp koff_tncl;    // [1/ms]
    fp kon_tncl;     // [1/mM/ms]
    fp Bmax_TnChigh; // [mM]
    fp koff_tnchca;  // [1/ms]
    fp kon_tnchca;   // [1/mM/ms]
    fp koff_tnchmg;  // [1/ms]
    fp kon_tnchmg;   // [1/mM/ms]
    fp Bmax_myosin;  // [mM]
    fp koff_myoca;   // [1/ms]
    fp kon_myoca;    // [1/mM/ms]
    fp koff_myomg;   // [1/ms]
    fp kon_myomg;    // [1/mM/ms]
    fp Bmax_SR;      // [mM]
    fp koff_sr;      // [1/ms]
    fp kon_sr;       // [1/mM/ms]
    fp Bmax_SLlowsl; // [mM]
    fp Bmax_SLlowj;  // [mM]
    fp koff_sll;     // [1/ms]
    fp kon_sll;      // [1/mM/ms]
    fp Bmax_SLhighsl;// [mM]
    fp Bmax_SLhighj; // [mM]
    fp koff_slh;     // [1/ms]
    fp kon_slh;      // [1/mM/ms]
    fp Bmax_Csqn;    // [mM]
    fp koff_csqn;    // [1/ms]
    fp kon_csqn;     // [1/mM/ms]

    // I_Na: Fast Na Current
    fp am;
    fp bm;
    fp ah;
    fp bh;
    fp aj;
    fp bj;
    fp I_Na_junc;
    fp I_Na_sl;

    // I_nabk: Na Background Current
    fp I_nabk_junc;
    fp I_nabk_sl;

    // I_nak: Na/K Pump Current
    fp sigma;
    fp fnak;
    fp I_nak_junc;
    fp I_nak_sl;
    fp I_nak;

    // I_kr: Rapidly Activating K Current
    fp gkr;
    fp xrss;
    fp tauxr;
    fp rkr;
    fp I_kr;

    // I_ks: Slowly Activating K Current
    fp pcaks_junc;
    fp pcaks_sl;
    fp gks_junc;
    fp gks_sl;
    fp eks;
    fp xsss;
    fp tauxs;
    fp I_ks_junc;
    fp I_ks_sl;
    fp I_ks;

    // I_kp: Plateau K current
    fp kp_kp;
    fp I_kp_junc;
    fp I_kp_sl;
    fp I_kp;

    // I_to: Transient Outward K Current (slow and fast components)
    fp xtoss;
    fp ytoss;
    fp rtoss;
    fp tauxtos;
    fp tauytos;
    fp taurtos;
    fp I_tos;

    fp tauxtof;
    fp tauytof;
    fp I_tof;
    fp I_to;

    // I_ki: Time-Independent K Current
    fp aki;
    fp bki;
    fp kiss;
    fp I_ki;

    // I_ClCa: Ca-activated Cl Current, I_Clbk: background Cl Current
    fp I_ClCa_junc;
    fp I_ClCa_sl;
    fp I_ClCa;
    fp I_Clbk;

    // I_Ca: L-type Calcium Current
    fp dss;
    fp taud;
    fp fss;
    fp tauf;

    fp ibarca_j;
    fp ibarca_sl;
    fp ibark;
    fp ibarna_j;
    fp ibarna_sl;
    fp I_Ca_junc;
    fp I_Ca_sl;
    fp I_Ca;
    fp I_CaK;
    fp I_CaNa_junc;
    fp I_CaNa_sl;

    // I_ncx: Na/Ca Exchanger flux
    fp Ka_junc;
    fp Ka_sl;
    fp s1_junc;
    fp s1_sl;
    fp s2_junc;
    fp s3_junc;
    fp s2_sl;
    fp s3_sl;
    fp I_ncx_junc;
    fp I_ncx_sl;
    fp I_ncx;

    // I_pca: Sarcolemmal Ca Pump Current
    fp I_pca_junc;
    fp I_pca_sl;
    fp I_pca;

    // I_cabk: Ca Background Current
    fp I_cabk_junc;
    fp I_cabk_sl;
    fp I_cabk;

    // SR fluxes: Calcium Release, SR Ca pump, SR Ca leak
    fp MaxSR;
    fp MinSR;
    fp kCaSR;
    fp koSRCa;
    fp kiSRCa;
    fp RI;
    fp J_SRCarel; // [mM/ms]
    fp J_serca;
    fp J_SRleak; // [mM/ms]

    // Cytosolic Ca Buffers
    fp J_CaB_cytosol;

    // Junctional and SL Ca Buffers
    fp J_CaB_junction;
    fp J_CaB_sl;

    // SR Ca Concentrations
    fp oneovervsr;

    // Sodium Concentrations
    fp I_Na_tot_junc; // [uA/uF]
    fp I_Na_tot_sl;   // [uA/uF]
    fp oneovervsl;

    // Potassium Concentration
    fp I_K_tot;

    // Calcium Concentrations
    fp I_Ca_tot_junc; // [uA/uF]
    fp I_Ca_tot_sl;   // [uA/uF]

    // Simulation type
    int state; // 0-none; 1-pace; 2-vclamp
    fp I_app;
    fp V_hold;
    fp V_test;
    fp V_clamp;
    fp R_clamp;

    // Membrane Potential
    fp I_Na_tot; // [uA/uF]
    fp I_Cl_tot; // [uA/uF]
    fp I_Ca_tot;
    fp I_tot;

    //=====================================================================
    //  EXECUTION
    //=====================================================================

    // input parameters
    cycleLength = __ldg(&d_params[15]);

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
    offset_16 = valu_offset + 15;
    offset_17 = valu_offset + 16;
    offset_18 = valu_offset + 17;
    offset_19 = valu_offset + 18;
    offset_20 = valu_offset + 19;
    offset_21 = valu_offset + 20;
    offset_22 = valu_offset + 21;
    offset_23 = valu_offset + 22;
    offset_24 = valu_offset + 23;
    offset_25 = valu_offset + 24;
    offset_26 = valu_offset + 25;
    offset_27 = valu_offset + 26;
    offset_28 = valu_offset + 27;
    offset_29 = valu_offset + 28;
    offset_30 = valu_offset + 29;
    offset_31 = valu_offset + 30;
    offset_32 = valu_offset + 31;
    offset_33 = valu_offset + 32;
    offset_34 = valu_offset + 33;
    offset_35 = valu_offset + 34;
    offset_36 = valu_offset + 35;
    offset_37 = valu_offset + 36;
    offset_38 = valu_offset + 37;
    offset_39 = valu_offset + 38;
    offset_40 = valu_offset + 39;
    offset_41 = valu_offset + 40;
    offset_42 = valu_offset + 41;
    offset_43 = valu_offset + 42;
    offset_44 = valu_offset + 43;
    offset_45 = valu_offset + 44;
    offset_46 = valu_offset + 45;

    // stored input array (use read-only cache)
    d_initvalu_1  = __ldg(&d_initvalu[offset_1]);
    d_initvalu_2  = __ldg(&d_initvalu[offset_2]);
    d_initvalu_3  = __ldg(&d_initvalu[offset_3]);
    d_initvalu_4  = __ldg(&d_initvalu[offset_4]);
    d_initvalu_5  = __ldg(&d_initvalu[offset_5]);
    d_initvalu_6  = __ldg(&d_initvalu[offset_6]);
    d_initvalu_7  = __ldg(&d_initvalu[offset_7]);
    d_initvalu_8  = __ldg(&d_initvalu[offset_8]);
    d_initvalu_9  = __ldg(&d_initvalu[offset_9]);
    d_initvalu_10 = __ldg(&d_initvalu[offset_10]);
    d_initvalu_11 = __ldg(&d_initvalu[offset_11]);
    d_initvalu_12 = __ldg(&d_initvalu[offset_12]);
    d_initvalu_13 = __ldg(&d_initvalu[offset_13]);
    d_initvalu_14 = __ldg(&d_initvalu[offset_14]);
    d_initvalu_15 = __ldg(&d_initvalu[offset_15]);
    d_initvalu_16 = __ldg(&d_initvalu[offset_16]);
    d_initvalu_17 = __ldg(&d_initvalu[offset_17]);
    d_initvalu_18 = __ldg(&d_initvalu[offset_18]);
    d_initvalu_19 = __ldg(&d_initvalu[offset_19]);
    d_initvalu_20 = __ldg(&d_initvalu[offset_20]);
    d_initvalu_21 = __ldg(&d_initvalu[offset_21]);
    // d_initvalu_22 = __ldg(&d_initvalu[offset_22]);
    d_initvalu_23 = __ldg(&d_initvalu[offset_23]);
    d_initvalu_24 = __ldg(&d_initvalu[offset_24]);
    d_initvalu_25 = __ldg(&d_initvalu[offset_25]);
    d_initvalu_26 = __ldg(&d_initvalu[offset_26]);
    d_initvalu_27 = __ldg(&d_initvalu[offset_27]);
    d_initvalu_28 = __ldg(&d_initvalu[offset_28]);
    d_initvalu_29 = __ldg(&d_initvalu[offset_29]);
    d_initvalu_30 = __ldg(&d_initvalu[offset_30]);
    d_initvalu_31 = __ldg(&d_initvalu[offset_31]);
    d_initvalu_32 = __ldg(&d_initvalu[offset_32]);
    d_initvalu_33 = __ldg(&d_initvalu[offset_33]);
    d_initvalu_34 = __ldg(&d_initvalu[offset_34]);
    d_initvalu_35 = __ldg(&d_initvalu[offset_35]);
    d_initvalu_36 = __ldg(&d_initvalu[offset_36]);
    d_initvalu_37 = __ldg(&d_initvalu[offset_37]);
    d_initvalu_38 = __ldg(&d_initvalu[offset_38]);
    d_initvalu_39 = __ldg(&d_initvalu[offset_39]);
    d_initvalu_40 = __ldg(&d_initvalu[offset_40]);
    // d_initvalu_41 = __ldg(&d_initvalu[offset_41]);
    // d_initvalu_42 = __ldg(&d_initvalu[offset_42]);
    // d_initvalu_43 = __ldg(&d_initvalu[offset_43]);
    // d_initvalu_44 = __ldg(&d_initvalu[offset_44]);
    // d_initvalu_45 = __ldg(&d_initvalu[offset_45]);
    // d_initvalu_46 = __ldg(&d_initvalu[offset_46]);

    // precompute some reused terms
    const fp V39 = d_initvalu_39;
    const fp FoRT_local_den = (fp)1.0; // for clarity

    // matlab constants undefined in c
    pi = (fp)3.1416;

    // Constants
    R    = (fp)8314.0;       // [J/kmol*K]
    Frdy = (fp)96485.0;      // [C/mol]
    Temp = (fp)310.0;        // [K]
    FoRT = Frdy / (R * Temp);
    Cmem = (fp)1.3810e-10;   // [F]
    Qpow = (Temp - (fp)310.0) / (fp)10.0;

    // Cell geometry
    cellLength = (fp)100.0;    // [um]
    cellRadius = (fp)10.25;    // [um]
    {
        const fp r2 = cellRadius * cellRadius;
        Vcell = pi * r2 * cellLength * (fp)1e-15; // [L]
    }
    Vmyo  = (fp)0.65  * Vcell;
    Vsr   = (fp)0.035 * Vcell;
    Vsl   = (fp)0.02  * Vcell;
    Vjunc = (fp)0.0539 * (fp)0.01 * Vcell;
    J_ca_juncsl = (fp)1.0 / (fp)1.2134e12;             // [L/msec]
    J_ca_slmyo  = (fp)1.0 / (fp)2.68510e11;            // [L/msec]
    J_na_juncsl = (fp)1.0 / ((fp)1.6382e12 / (fp)3.0 * (fp)100.0); // [L/msec]
    J_na_slmyo  = (fp)1.0 / ((fp)1.8308e10 / (fp)3.0 * (fp)100.0); // [L/msec]

    // Fractional currents in compartments
    Fjunc     = (fp)0.11;
    Fsl       = (fp)1.0 - Fjunc;
    Fjunc_CaL = (fp)0.9;
    Fsl_CaL   = (fp)1.0 - Fjunc_CaL;

    // Fixed ion concentrations
    Cli = (fp)15.0;
    Clo = (fp)150.0;
    Ko  = (fp)5.4;
    Nao = (fp)140.0;
    Cao = (fp)1.8;
    Mgi = (fp)1.0;

    // Nernst Potentials
    ena_junc = ((fp)1.0 / FoRT) * log(Nao / d_initvalu_32);
    ena_sl   = ((fp)1.0 / FoRT) * log(Nao / d_initvalu_33);
    ek       = ((fp)1.0 / FoRT) * log(Ko / d_initvalu_35);
    eca_junc = ((fp)1.0 / (FoRT * (fp)2.0)) * log(Cao / d_initvalu_36);
    eca_sl   = ((fp)1.0 / (FoRT * (fp)2.0)) * log(Cao / d_initvalu_37);
    ecl      = ((fp)1.0 / FoRT) * log(Cli / Clo);

    // Na transport parameters
    GNa     = (fp)16.0;
    GNaB    = (fp)0.297e-3;
    IbarNaK = (fp)1.90719;
    KmNaip  = (fp)11.0;
    KmKo    = (fp)1.5;

    // K current parameters
    pNaK    = (fp)0.01833;
    GtoSlow = (fp)0.06;
    GtoFast = (fp)0.02;
    gkp     = (fp)0.001;

    // Cl current parameters
    GClCa  = (fp)0.109625;
    GClB   = (fp)9e-3;
    KdClCa = (fp)100e-3;

    // I_Ca parameters
    pNa    = (fp)1.5e-8;
    pCa    = (fp)5.4e-4;
    pK     = (fp)2.7e-7;
    Q10CaL = (fp)1.8;

    // Ca transport parameters
    IbarNCX   = (fp)9.0;
    KmCai     = (fp)3.59e-3;
    KmCao     = (fp)1.3;
    KmNai     = (fp)12.29;
    KmNao     = (fp)87.5;
    ksat      = (fp)0.27;
    nu        = (fp)0.35;
    Kdact     = (fp)0.256e-3;
    Q10NCX    = (fp)1.57;
    IbarSLCaP = (fp)0.0673;
    KmPCa     = (fp)0.5e-3;
    GCaB      = (fp)2.513e-4;
    Q10SLCaP  = (fp)2.35;

    // SR flux parameters
    Q10SRCaP   = (fp)2.6;
    Vmax_SRCaP = (fp)2.86e-4;
    Kmf        = (fp)0.246e-3;
    Kmr        = (fp)1.7;
    hillSRCaP  = (fp)1.787;
    ks         = (fp)25.0;
    koCa       = (fp)10.0;
    kom        = (fp)0.06;
    kiCa       = (fp)0.5;
    kim        = (fp)0.005;
    ec50SR     = (fp)0.45;

    // Buffering parameters
    Bmax_Naj     = (fp)7.561;
    Bmax_Nasl    = (fp)1.65;
    koff_na      = (fp)1e-3;
    kon_na       = (fp)0.1e-3;
    Bmax_TnClow  = (fp)70e-3;
    koff_tncl    = (fp)19.6e-3;
    kon_tncl     = (fp)32.7;
    Bmax_TnChigh = (fp)140e-3;
    koff_tnchca  = (fp)0.032e-3;
    kon_tnchca   = (fp)2.37;
    koff_tnchmg  = (fp)3.33e-3;
    kon_tnchmg   = (fp)3e-3;
    Bmax_myosin  = (fp)140e-3;
    koff_myoca   = (fp)0.46e-3;
    kon_myoca    = (fp)13.8;
    koff_myomg   = (fp)0.057e-3;
    kon_myomg    = (fp)0.0157;
    Bmax_SR      = (fp)19.0 * (fp)0.9e-3;
    koff_sr      = (fp)60e-3;
    kon_sr       = (fp)100.0;
    Bmax_SLlowsl = (fp)37.38e-3 * Vmyo / Vsl;
    Bmax_SLlowj  = (fp)4.62e-3 * Vmyo / Vjunc * (fp)0.1;
    koff_sll     = (fp)1300e-3;
    kon_sll      = (fp)100.0;
    Bmax_SLhighsl= (fp)13.35e-3 * Vmyo / Vsl;
    Bmax_SLhighj = (fp)1.65e-3 * Vmyo / Vjunc * (fp)0.1;
    koff_slh     = (fp)30e-3;
    kon_slh      = (fp)100.0;
    Bmax_Csqn    = (fp)2.7;
    koff_csqn    = (fp)65.0;
    kon_csqn     = (fp)100.0;

    // I_Na: Fast Na Current
    const fp V39p47_13 = V39 + (fp)47.13;
    am = (fp)0.32 * V39p47_13 / ((fp)1.0 - expf_rn((fp)-0.1 * V39p47_13));
    bm = (fp)0.08 * expf_rn(-V39 / (fp)11.0);

    if (V39 >= (fp)-40.0) {
        ah = (fp)0.0;
        aj = (fp)0.0;
        const fp num_bh = (fp)1.0;
        const fp den_bh = (fp)0.13 * ((fp)1.0 + expf_rn(-(V39 + (fp)10.66) / (fp)11.1));
        bh = num_bh / den_bh;
        bj = (fp)0.3 * expf_rn((fp)-2.535e-7 * V39) /
             ((fp)1.0 + expf_rn((fp)-0.1 * (V39 + (fp)32.0)));
    } else {
        ah = (fp)0.135 * expf_rn(((fp)80.0 + V39) / (fp)-6.8);
        bh = (fp)3.56 * expf_rn((fp)0.079 * V39) +
             (fp)3.1e5 * expf_rn((fp)0.35 * V39);
        aj = (fp)(-127140.0 * expf_rn((fp)0.2444 * V39) -
                  (fp)3.474e-5 * expf_rn((fp)-0.04391 * V39)) *
             (V39 + (fp)37.78) /
             ((fp)1.0 + expf_rn((fp)0.311 * (V39 + (fp)79.23)));
        bj = (fp)0.1212 * expf_rn((fp)-0.01052 * V39) /
             ((fp)1.0 + expf_rn((fp)-0.1378 * (V39 + (fp)40.14)));
    }

    d_finavalu[offset_1] = am * ((fp)1.0 - d_initvalu_1) - bm * d_initvalu_1;
    d_finavalu[offset_2] = ah * ((fp)1.0 - d_initvalu_2) - bh * d_initvalu_2;
    d_finavalu[offset_3] = aj * ((fp)1.0 - d_initvalu_3) - bj * d_initvalu_3;

    {
        const fp m3 = d_initvalu_1 * d_initvalu_1 * d_initvalu_1;
        const fp h  = d_initvalu_2;
        const fp jg = d_initvalu_3;
        const fp V_enaj = V39 - ena_junc;
        const fp V_enas = V39 - ena_sl;
        I_Na_junc = Fjunc * GNa * m3 * h * jg * V_enaj;
        I_Na_sl   = Fsl   * GNa * m3 * h * jg * V_enas;
    }

    // I_nabk: Na Background Current
    I_nabk_junc = Fjunc * GNaB * (V39 - ena_junc);
    I_nabk_sl   = Fsl   * GNaB * (V39 - ena_sl);

    // I_nak: Na/K Pump Current
    sigma = (expf_rn(Nao / (fp)67.3) - (fp)1.0) / (fp)7.0;
    fnak = (fp)1.0 / ((fp)1.0 +
                      (fp)0.1245 * expf_rn((fp)-0.1 * V39 * FoRT) +
                      (fp)0.0365 * sigma * expf_rn(-V39 * FoRT));
    {
        const fp KmNaip_4 = KmNaip / d_initvalu_32;
        const fp KmNaip_4_4 = KmNaip_4 * KmNaip_4 * KmNaip_4 * KmNaip_4;
        I_nak_junc = Fjunc * IbarNaK * fnak * Ko /
                     ((fp)1.0 + KmNaip_4_4) / (Ko + KmKo);
    }
    {
        const fp KmNaip_4s = KmNaip / d_initvalu_33;
        const fp KmNaip_4s_4 = KmNaip_4s * KmNaip_4s * KmNaip_4s * KmNaip_4s;
        I_nak_sl = Fsl * IbarNaK * fnak * Ko /
                   ((fp)1.0 + KmNaip_4s_4) / (Ko + KmKo);
    }
    I_nak = I_nak_junc + I_nak_sl;

    // I_kr: Rapidly Activating K Current
    gkr  = (fp)0.03 * sqrtf_rn(Ko / (fp)5.4);
    xrss = (fp)1.0 / ((fp)1.0 + expf_rn(-(V39 + (fp)50.0) / (fp)7.5));
    tauxr = (fp)1.0 /
            ((fp)0.00138 * (V39 + (fp)7.0) /
                 ((fp)1.0 - expf_rn((fp)-0.123 * (V39 + (fp)7.0))) +
             (fp)6.1e-4 * (V39 + (fp)10.0) /
                 (expf_rn((fp)0.145 * (V39 + (fp)10.0)) - (fp)1.0));
    d_finavalu[offset_12] = (xrss - d_initvalu_12) / tauxr;
    rkr = (fp)1.0 / ((fp)1.0 + expf_rn((V39 + (fp)33.0) / (fp)22.4));
    I_kr = gkr * d_initvalu_12 * rkr * (V39 - ek);

    // I_ks: Slowly Activating K Current
    pcaks_junc = (fp)-log10(d_initvalu_36) + (fp)3.0;
    pcaks_sl   = (fp)-log10(d_initvalu_37) + (fp)3.0;
    gks_junc   = (fp)0.07 * ((fp)0.057 + (fp)0.19 /
                 ((fp)1.0 + expf_rn(((-7.2f + pcaks_junc) / (fp)0.6))));
    gks_sl     = (fp)0.07 * ((fp)0.057 + (fp)0.19 /
                 ((fp)1.0 + expf_rn(((-7.2f + pcaks_sl) / (fp)0.6))));
    eks        = ((fp)1.0 / FoRT) *
                 log((Ko + pNaK * Nao) / (d_initvalu_35 + pNaK * d_initvalu_34));
    xsss = (fp)1.0 / ((fp)1.0 + expf_rn(-(V39 - (fp)1.5) / (fp)16.7));
    tauxs = (fp)1.0 /
            ((fp)7.19e-5 * (V39 + (fp)30.0) /
                 ((fp)1.0 - expf_rn((fp)-0.148 * (V39 + (fp)30.0))) +
             (fp)1.31e-4 * (V39 + (fp)30.0) /
                 (expf_rn((fp)0.0687 * (V39 + (fp)30.0)) - (fp)1.0));
    d_finavalu[offset_13] = (xsss - d_initvalu_13) / tauxs;
    {
        const fp xs2 = d_initvalu_12 * d_initvalu_12;
        I_ks_junc = Fjunc * gks_junc * xs2 * (V39 - eks);
    }
    {
        const fp xs2s = d_initvalu_13 * d_initvalu_13;
        I_ks_sl = Fsl * gks_sl * xs2s * (V39 - eks);
    }
    I_ks = I_ks_junc + I_ks_sl;

    // I_kp: Plateau K current
    kp_kp   = (fp)1.0 / ((fp)1.0 + expf_rn((fp)7.488 - V39 / (fp)5.98));
    I_kp_junc = Fjunc * gkp * kp_kp * (V39 - ek);
    I_kp_sl   = Fsl   * gkp * kp_kp * (V39 - ek);
    I_kp      = I_kp_junc + I_kp_sl;

    // I_to: Transient Outward K Current
    xtoss   = (fp)1.0 / ((fp)1.0 + expf_rn(-(V39 + (fp)3.0) / (fp)15.0));
    ytoss   = (fp)1.0 / ((fp)1.0 + expf_rn((V39 + (fp)33.5) / (fp)10.0));
    rtoss   = ytoss;
    tauxtos = (fp)9.0 / ((fp)1.0 + expf_rn((V39 + (fp)3.0) / (fp)15.0)) + (fp)0.5;
    tauytos = (fp)3e3 / ((fp)1.0 + expf_rn((V39 + (fp)60.0) / (fp)10.0)) + (fp)30.0;
    taurtos = (fp)2800.0 / ((fp)1.0 + expf_rn((V39 + (fp)60.0) / (fp)10.0)) + (fp)220.0;
    d_finavalu[offset_8]  = (xtoss - d_initvalu_8)  / tauxtos;
    d_finavalu[offset_9]  = (ytoss - d_initvalu_9)  / tauytos;
    d_finavalu[offset_40] = (rtoss - d_initvalu_40) / taurtos;
    I_tos = GtoSlow * d_initvalu_8 * (d_initvalu_9 + (fp)0.5 * d_initvalu_40) *
            (V39 - ek);

    tauxtof = (fp)3.5 * expf_rn(-V39 * V39 / ((fp)30.0 * (fp)30.0)) + (fp)1.5;
    tauytof = (fp)20.0 / ((fp)1.0 + expf_rn((V39 + (fp)33.5) / (fp)10.0)) + (fp)20.0;
    d_finavalu[offset_10] = (xtoss - d_initvalu_10) / tauxtof;
    d_finavalu[offset_11] = (ytoss - d_initvalu_11) / tauytof;
    I_tof = GtoFast * d_initvalu_10 * d_initvalu_11 * (V39 - ek);
    I_to  = I_tos + I_tof;

    // I_ki: Time-Independent K Current
    aki = (fp)1.02 / ((fp)1.0 + expf_rn((fp)0.2385 * (V39 - ek - (fp)59.215)));
    bki = ((fp)0.49124 * expf_rn((fp)0.08032 * (V39 + (fp)5.476 - ek)) +
           expf_rn((fp)0.06175 * (V39 - ek - (fp)594.31))) /
          ((fp)1.0 + expf_rn((fp)-0.5143 * (V39 - ek + (fp)4.753)));
    kiss = aki / (aki + bki);
    I_ki = (fp)0.9 * sqrtf_rn(Ko / (fp)5.4) * kiss * (V39 - ek);

    // I_ClCa and I_Clbk
    I_ClCa_junc =
        Fjunc * GClCa / ((fp)1.0 + KdClCa / d_initvalu_36) * (V39 - ecl);
    I_ClCa_sl =
        Fsl * GClCa / ((fp)1.0 + KdClCa / d_initvalu_37) * (V39 - ecl);
    I_ClCa = I_ClCa_junc + I_ClCa_sl;
    I_Clbk = GClB * (V39 - ecl);

    // I_Ca: L-type Calcium Current
    dss = (fp)1.0 / ((fp)1.0 + expf_rn(-(V39 + (fp)14.5) / (fp)6.0));
    {
        const fp num_taud = dss * ((fp)1.0 - expf_rn(-(V39 + (fp)14.5) / (fp)6.0));
        const fp den_taud = (fp)0.035 * (V39 + (fp)14.5);
        taud = num_taud / den_taud;
    }
    fss = (fp)1.0 / ((fp)1.0 + expf_rn((V39 + (fp)35.06) / (fp)3.6)) +
          (fp)0.6 / ((fp)1.0 + expf_rn(((fp)50.0 - V39) / (fp)20.0));
    {
        const fp x = (fp)0.0337 * (V39 + (fp)14.5);
        tauf = (fp)1.0 / ((fp)0.0197 * expf_rn(-x * x) + (fp)0.02);
    }
    d_finavalu[offset_4] = (dss - d_initvalu_4) / taud;
    d_finavalu[offset_5] = (fss - d_initvalu_5) / tauf;
    d_finavalu[offset_6] = (fp)1.7 * d_initvalu_36 * ((fp)1.0 - d_initvalu_6) -
                           (fp)11.9e-3 * d_initvalu_6;
    d_finavalu[offset_7] = (fp)1.7 * d_initvalu_37 * ((fp)1.0 - d_initvalu_7) -
                           (fp)11.9e-3 * d_initvalu_7;

    {
        const fp VFoRT2 = V39 * Frdy * FoRT;
        const fp exp2V  = expf_rn((fp)2.0 * V39 * FoRT);
        const fp num_j =
            pCa * (fp)4.0 * VFoRT2 *
            ((fp)0.341 * d_initvalu_36 * exp2V - (fp)0.341 * Cao);
        const fp den_j = exp2V - (fp)1.0;
        ibarca_j = num_j / den_j;

        const fp num_sl =
            pCa * (fp)4.0 * VFoRT2 *
            ((fp)0.341 * d_initvalu_37 * exp2V - (fp)0.341 * Cao);
        ibarca_sl = num_sl / den_j;

        const fp expV = expf_rn(V39 * FoRT);
        const fp num_k =
            pK * VFoRT2 *
            ((fp)0.75 * d_initvalu_35 * expV - (fp)0.75 * Ko);
        const fp den_k = expV - (fp)1.0;
        ibark = num_k / den_k;

        const fp num_na_j =
            pNa * VFoRT2 *
            ((fp)0.75 * d_initvalu_32 * expV - (fp)0.75 * Nao);
        const fp num_na_sl =
            pNa * VFoRT2 *
            ((fp)0.75 * d_initvalu_33 * expV - (fp)0.75 * Nao);
        const fp den_na = den_k;
        ibarna_j  = num_na_j  / den_na;
        ibarna_sl = num_na_sl / den_na;
    }

    {
        const fp open_j = (fp)1.0 - d_initvalu_6;
        const fp open_sl = (fp)1.0 - d_initvalu_7;
        const fp QCaL = powf(Q10CaL, Qpow) * (fp)0.45;
        const fp d4f = d_initvalu_4 * d_initvalu_5;
        I_Ca_junc = Fjunc_CaL * ibarca_j  * d4f * open_j  * QCaL;
        I_Ca_sl   = Fsl_CaL   * ibarca_sl * d4f * open_sl * QCaL;
        I_Ca = I_Ca_junc + I_Ca_sl;
        d_finavalu[offset_43] = -I_Ca * Cmem / (Vmyo * (fp)2.0 * Frdy) * (fp)1e3;

        const fp gate_sum = Fjunc_CaL * open_j + Fsl_CaL * open_sl;
        I_CaK = ibark * d4f * gate_sum * QCaL;
        I_CaNa_junc = Fjunc_CaL * ibarna_j  * d4f * open_j  * QCaL;
        I_CaNa_sl   = Fsl_CaL   * ibarna_sl * d4f * open_sl * QCaL;
    }

    // I_ncx: Na/Ca Exchanger flux
    {
        const fp Kd_over_Caj = Kdact / d_initvalu_36;
        const fp Kd_over_Casl = Kdact / d_initvalu_37;
        const fp Kd_over_Caj3  = Kd_over_Caj  * Kd_over_Caj  * Kd_over_Caj;
        const fp Kd_over_Casl3 = Kd_over_Casl * Kd_over_Casl * Kd_over_Casl;
        Ka_junc = (fp)1.0 / ((fp)1.0 + Kd_over_Caj3);
        Ka_sl   = (fp)1.0 / ((fp)1.0 + Kd_over_Casl3);
    }
    {
        const fp Nai_j = d_initvalu_32;
        const fp Nai_sl = d_initvalu_33;
        const fp Cai_j = d_initvalu_36;
        const fp Cai_sl = d_initvalu_37;
        const fp exp_nuV = expf_rn(nu * V39 * FoRT);
        const fp exp_nu1V = expf_rn((nu - (fp)1.0) * V39 * FoRT);

        const fp Nai_j3  = Nai_j  * Nai_j  * Nai_j;
        const fp Nai_sl3 = Nai_sl * Nai_sl * Nai_sl;
        const fp Nao3    = Nao * Nao * Nao;

        s1_junc = exp_nuV  * Nai_j3  * Cao;
        s1_sl   = exp_nuV  * Nai_sl3 * Cao;
        s2_junc = exp_nu1V * Nao3    * Cai_j;
        s2_sl   = exp_nu1V * Nao3    * Cai_sl;

        const fp KmNao3 = KmNao * KmNao * KmNao;
        const fp KmNai3 = KmNai * KmNai * KmNai;
        const fp Nai_j3_term  = Nai_j3  * Cao;
        const fp Nai_sl3_term = Nai_sl3 * Cao;
        const fp Nao3Caj      = Nao3 * Cai_j;
        const fp Nao3Casl     = Nao3 * Cai_sl;

        const fp term_common_j =
            KmCai * Nao3 * ((fp)1.0 + (Nai_j3  / (KmNai3))) +
            KmNao3 * Cai_j +
            KmNai3 * Cao * ((fp)1.0 + Cai_j / KmCai) +
            KmCao * Nai_j3 +
            Nai_j3_term +
            Nao3Caj;

        const fp term_common_sl =
            KmCai * Nao3 * ((fp)1.0 + (Nai_sl3 / (KmNai3))) +
            KmNao3 * Cai_sl +
            KmNai3 * Cao * ((fp)1.0 + Cai_sl / KmCai) +
            KmCao * Nai_sl3 +
            Nai_sl3_term +
            Nao3Casl;

        const fp ksat_exp = (fp)1.0 + ksat * exp_nu1V;
        s3_junc = term_common_j  * ksat_exp;
        s3_sl   = term_common_sl * ksat_exp;
    }
    {
        const fp QNCX = powf(Q10NCX, Qpow);
        I_ncx_junc = Fjunc * IbarNCX * QNCX * Ka_junc *
                     (s1_junc - s2_junc) / s3_junc;
        I_ncx_sl   = Fsl   * IbarNCX * QNCX * Ka_sl *
                     (s1_sl   - s2_sl)   / s3_sl;
        I_ncx = I_ncx_junc + I_ncx_sl;
        d_finavalu[offset_45] = (fp)2.0 * I_ncx * Cmem / (Vmyo * (fp)2.0 * Frdy) * (fp)1e3;
    }

    // I_pca: Sarcolemmal Ca Pump Current
    {
        const fp QSL = powf(Q10SLCaP, Qpow);
        const fp Cai_j = d_initvalu_36;
        const fp Cai_sl = d_initvalu_37;
        const fp powCa_j  = powf(Cai_j,  (fp)1.6);
        const fp powCa_sl = powf(Cai_sl, (fp)1.6);
        const fp powKm = powf(KmPCa, (fp)1.6);

        I_pca_junc = Fjunc * QSL * IbarSLCaP * powCa_j /
                     (powKm + powCa_j);
        I_pca_sl   = Fsl   * QSL * IbarSLCaP * powCa_sl /
                     (powKm + powCa_sl);
        I_pca = I_pca_junc + I_pca_sl;
        d_finavalu[offset_44] = -I_pca * Cmem / (Vmyo * (fp)2.0 * Frdy) * (fp)1e3;
    }

    // I_cabk: Ca Background Current
    I_cabk_junc = Fjunc * GCaB * (V39 - eca_junc);
    I_cabk_sl   = Fsl   * GCaB * (V39 - eca_sl);
    I_cabk      = I_cabk_junc + I_cabk_sl;
    d_finavalu[offset_46] = -I_cabk * Cmem / (Vmyo * (fp)2.0 * Frdy) * (fp)1e3;

    // SR fluxes: Calcium Release, SR Ca pump, SR Ca leak
    MaxSR = (fp)15.0;
    MinSR = (fp)1.0;
    {
        const fp ec_over_Casr = ec50SR / d_initvalu_31;
        const fp ec_over_Casr_25 = powf(ec_over_Casr, (fp)2.5);
        kCaSR = MaxSR - (MaxSR - MinSR) / ((fp)1.0 + ec_over_Casr_25);
    }
    koSRCa = koCa / kCaSR;
    kiSRCa = kiCa * kCaSR;
    RI = (fp)1.0 - d_initvalu_14 - d_initvalu_15 - d_initvalu_16;

    {
        const fp Ca_j = d_initvalu_36;
        const fp Ca_j2 = Ca_j * Ca_j;
        const fp R_state = d_initvalu_14;
        const fp O_state = d_initvalu_15;
        const fp I_state = d_initvalu_16;

        d_finavalu[offset_14] =
            (kim * RI - kiSRCa * Ca_j * R_state) -
            (koSRCa * Ca_j2 * R_state - kom * O_state);
        d_finavalu[offset_15] =
            (koSRCa * Ca_j2 * R_state - kom * O_state) -
            (kiSRCa * Ca_j * O_state - kim * I_state);
        d_finavalu[offset_16] =
            (kiSRCa * Ca_j * O_state - kim * I_state) -
            (kom * I_state - koSRCa * Ca_j2 * RI);
        J_SRCarel = ks * O_state * (d_initvalu_31 - Ca_j);
    }

    {
        const fp Ca_i  = d_initvalu_38;
        const fp Ca_sr = d_initvalu_31;
        const fp Ca_i_over_Kmf  = Ca_i  / Kmf;
        const fp Ca_sr_over_Kmr = Ca_sr / Kmr;
        const fp Ca_i_pow  = powf(Ca_i_over_Kmf,  hillSRCaP);
        const fp Ca_sr_pow = powf(Ca_sr_over_Kmr, hillSRCaP);
        J_serca = powf(Q10SRCaP, Qpow) * Vmax_SRCaP *
                  (Ca_i_pow - Ca_sr_pow) /
                  ((fp)1.0 + Ca_i_pow + Ca_sr_pow);
    }
    J_SRleak = (fp)5.348e-6 * (d_initvalu_31 - d_initvalu_36);

    // Sodium and Calcium Buffering
    d_finavalu[offset_17] =
        kon_na * d_initvalu_32 * (Bmax_Naj - d_initvalu_17) -
        koff_na * d_initvalu_17;
    d_finavalu[offset_18] =
        kon_na * d_initvalu_33 * (Bmax_Nasl - d_initvalu_18) -
        koff_na * d_initvalu_18;

    // Cytosolic Ca Buffers
    d_finavalu[offset_19] =
        kon_tncl * d_initvalu_38 * (Bmax_TnClow - d_initvalu_19) -
        koff_tncl * d_initvalu_19;
    d_finavalu[offset_20] =
        kon_tnchca * d_initvalu_38 *
            (Bmax_TnChigh - d_initvalu_20 - d_initvalu_21) -
        koff_tnchca * d_initvalu_20;
    d_finavalu[offset_21] =
        kon_tnchmg * Mgi *
            (Bmax_TnChigh - d_initvalu_20 - d_initvalu_21) -
        koff_tnchmg * d_initvalu_21;
    d_finavalu[offset_22] = (fp)0.0;
    d_finavalu[offset_23] =
        kon_myoca * d_initvalu_38 *
            (Bmax_myosin - d_initvalu_23 - d_initvalu_24) -
        koff_myoca * d_initvalu_23;
    d_finavalu[offset_24] =
        kon_myomg * Mgi *
            (Bmax_myosin - d_initvalu_23 - d_initvalu_24) -
        koff_myomg * d_initvalu_24;
    d_finavalu[offset_25] =
        kon_sr * d_initvalu_38 * (Bmax_SR - d_initvalu_25) -
        koff_sr * d_initvalu_25;

    J_CaB_cytosol = d_finavalu[offset_19] + d_finavalu[offset_20] +
                    d_finavalu[offset_21] + d_finavalu[offset_22] +
                    d_finavalu[offset_23] + d_finavalu[offset_24] +
                    d_finavalu[offset_25];

    // Junctional and SL Ca Buffers
    d_finavalu[offset_26] =
        kon_sll * d_initvalu_36 * (Bmax_SLlowj - d_initvalu_26) -
        koff_sll * d_initvalu_26;
    d_finavalu[offset_27] =
        kon_sll * d_initvalu_37 * (Bmax_SLlowsl - d_initvalu_27) -
        koff_sll * d_initvalu_27;
    d_finavalu[offset_28] =
        kon_slh * d_initvalu_36 * (Bmax_SLhighj - d_initvalu_28) -
        koff_slh * d_initvalu_28;
    d_finavalu[offset_29] =
        kon_slh * d_initvalu_37 * (Bmax_SLhighsl - d_initvalu_29) -
        koff_slh * d_initvalu_29;

    J_CaB_junction = d_finavalu[offset_26] + d_finavalu[offset_28];
    J_CaB_sl       = d_finavalu[offset_27] + d_finavalu[offset_29];

    // SR Ca Concentrations
    d_finavalu[offset_30] =
        kon_csqn * d_initvalu_31 * (Bmax_Csqn - d_initvalu_30) -
        koff_csqn * d_initvalu_30;
    oneovervsr = (fp)1.0 / Vsr;
    d_finavalu[offset_31] =
        J_serca * Vmyo * oneovervsr -
        (J_SRleak * Vmyo * oneovervsr + J_SRCarel) -
        d_finavalu[offset_30];

    // Sodium Concentrations
    I_Na_tot_junc = I_Na_junc + I_nabk_junc + (fp)3.0 * I_ncx_junc +
                    (fp)3.0 * I_nak_junc + I_CaNa_junc;
    I_Na_tot_sl   = I_Na_sl   + I_nabk_sl   + (fp)3.0 * I_ncx_sl   +
                    (fp)3.0 * I_nak_sl   + I_CaNa_sl;

    d_finavalu[offset_32] =
        -I_Na_tot_junc * Cmem / (Vjunc * Frdy) +
        J_na_juncsl / Vjunc * (d_initvalu_33 - d_initvalu_32) -
        d_finavalu[offset_17];

    oneovervsl = (fp)1.0 / Vsl;
    d_finavalu[offset_33] =
        -I_Na_tot_sl * Cmem * oneovervsl / Frdy +
        J_na_juncsl * oneovervsl * (d_initvalu_32 - d_initvalu_33) +
        J_na_slmyo  * oneovervsl * (d_initvalu_34 - d_initvalu_33) -
        d_finavalu[offset_18];

    d_finavalu[offset_34] =
        J_na_slmyo / Vmyo * (d_initvalu_33 - d_initvalu_34);

    // Potassium Concentration
    I_K_tot = I_to + I_kr + I_ks + I_ki - (fp)2.0 * I_nak + I_CaK + I_kp;
    d_finavalu[offset_35] = (fp)0.0;

    // Calcium Concentrations
    I_Ca_tot_junc =
        I_Ca_junc + I_cabk_junc + I_pca_junc - (fp)2.0 * I_ncx_junc;
    I_Ca_tot_sl =
        I_Ca_sl + I_cabk_sl + I_pca_sl - (fp)2.0 * I_ncx_sl;

    d_finavalu[offset_36] =
        -I_Ca_tot_junc * Cmem / (Vjunc * (fp)2.0 * Frdy) +
        J_ca_juncsl / Vjunc * (d_initvalu_37 - d_initvalu_36) -
        J_CaB_junction +
        J_SRCarel * Vsr / Vjunc +
        J_SRleak * Vmyo / Vjunc;

    d_finavalu[offset_37] =
        -I_Ca_tot_sl * Cmem / (Vsl * (fp)2.0 * Frdy) +
        J_ca_juncsl / Vsl * (d_initvalu_36 - d_initvalu_37) +
        J_ca_slmyo / Vsl * (d_initvalu_38 - d_initvalu_37) -
        J_CaB_sl;

    d_finavalu[offset_38] =
        -J_serca - J_CaB_cytosol +
        J_ca_slmyo / Vmyo * (d_initvalu_37 - d_initvalu_38);

    // Simulation type
    state = 1;
    switch (state) {
    case 0:
        I_app = (fp)0.0;
        break;
    case 1:
        if (fmod(timeinst, cycleLength) <= (fp)5.0) {
            I_app = (fp)9.5;
        } else {
            I_app = (fp)0.0;
        }
        break;
    case 2:
        V_hold = (fp)-55.0;
        V_test = (fp)0.0;
        if (timeinst > (fp)0.5 && timeinst < (fp)200.5) {
            V_clamp = V_test;
        } else {
            V_clamp = V_hold;
        }
        R_clamp = (fp)0.04;
        I_app = (V_clamp - V39) / R_clamp;
        break;
    }

    // Membrane Potential
    I_Na_tot = I_Na_tot_junc + I_Na_tot_sl;
    I_Cl_tot = I_ClCa + I_Clbk;
    I_Ca_tot = I_Ca_tot_junc + I_Ca_tot_sl;
    I_tot    = I_Na_tot + I_Cl_tot + I_Ca_tot + I_K_tot;
    d_finavalu[offset_39] = -(I_tot - I_app);

    // Set unused output values to 0
    d_finavalu[offset_41] = (fp)0.0;
    d_finavalu[offset_42] = (fp)0.0;
}
