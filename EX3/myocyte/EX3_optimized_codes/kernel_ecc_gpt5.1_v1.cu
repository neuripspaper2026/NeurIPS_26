__device__ void kernel_ecc(fp timeinst, fp * __restrict__ d_initvalu, fp * __restrict__ d_finavalu,
                           int valu_offset, fp * __restrict__ d_params) {

    //=====================================================================
    //	VARIABLES
    //=====================================================================

    // input parameters
    fp cycleLength;

    // variable references				// GET VARIABLES FROM MEMORY AND SAVE LOCALLY
    // !!!!!!!!!!!!!!!!!!
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
    fp pNa; // [cm/sec]
    fp pCa; // [cm/sec]
    fp pK;  // [cm/sec]
    fp Q10CaL;

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
    fp Vmax_SRCaP; // [mM/msec] (mmol/L cytosol/msec)
    fp Kmf;        // [mM]
    fp Kmr;        // [mM]L cytosol
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
    fp Bmax_TnClow;  // [mM], TnC low affinity
    fp koff_tncl;    // [1/ms]
    fp kon_tncl;     // [1/mM/ms]
    fp Bmax_TnChigh; // [mM], TnC high affinity
    fp koff_tnchca;  // [1/ms]
    fp kon_tnchca;   // [1/mM/ms]
    fp koff_tnchmg;  // [1/ms]
    fp kon_tnchmg;   // [1/mM/ms]
    fp Bmax_myosin;  // [mM], Myosin buffering
    fp koff_myoca;   // [1/ms]
    fp kon_myoca;    // [1/mM/ms]
    fp koff_myomg;   // [1/ms]
    fp kon_myomg;    // [1/mM/ms]
    fp Bmax_SR;      // [mM]
    fp koff_sr;      // [1/ms]
    fp kon_sr;       // [1/mM/ms]
    fp Bmax_SLlowsl; // [mM], SL buffering
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
    fp J_SRleak; //   [mM/ms]

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

    //	Simulation type
    int state; // 0-none; 1-pace; 2-vclamp
    fp I_app;
    fp V_hold;
    fp V_test;
    fp V_clamp;
    fp R_clamp;

    //	Membrane Potential
    fp I_Na_tot; // [uA/uF]
    fp I_Cl_tot; // [uA/uF]
    fp I_Ca_tot;
    fp I_tot;

    //=====================================================================
    //	EXECUTION
    //=====================================================================

    // input parameters
    cycleLength = d_params[15];

    // variable references (hoist base index)
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

    // stored input array (use FMA-friendly loads; values reused heavily)
    d_initvalu_1  = d_initvalu[offset_1];
    d_initvalu_2  = d_initvalu[offset_2];
    d_initvalu_3  = d_initvalu[offset_3];
    d_initvalu_4  = d_initvalu[offset_4];
    d_initvalu_5  = d_initvalu[offset_5];
    d_initvalu_6  = d_initvalu[offset_6];
    d_initvalu_7  = d_initvalu[offset_7];
    d_initvalu_8  = d_initvalu[offset_8];
    d_initvalu_9  = d_initvalu[offset_9];
    d_initvalu_10 = d_initvalu[offset_10];
    d_initvalu_11 = d_initvalu[offset_11];
    d_initvalu_12 = d_initvalu[offset_12];
    d_initvalu_13 = d_initvalu[offset_13];
    d_initvalu_14 = d_initvalu[offset_14];
    d_initvalu_15 = d_initvalu[offset_15];
    d_initvalu_16 = d_initvalu[offset_16];
    d_initvalu_17 = d_initvalu[offset_17];
    d_initvalu_18 = d_initvalu[offset_18];
    d_initvalu_19 = d_initvalu[offset_19];
    d_initvalu_20 = d_initvalu[offset_20];
    d_initvalu_21 = d_initvalu[offset_21];
    // d_initvalu_22 = d_initvalu[offset_22];
    d_initvalu_23 = d_initvalu[offset_23];
    d_initvalu_24 = d_initvalu[offset_24];
    d_initvalu_25 = d_initvalu[offset_25];
    d_initvalu_26 = d_initvalu[offset_26];
    d_initvalu_27 = d_initvalu[offset_27];
    d_initvalu_28 = d_initvalu[offset_28];
    d_initvalu_29 = d_initvalu[offset_29];
    d_initvalu_30 = d_initvalu[offset_30];
    d_initvalu_31 = d_initvalu[offset_31];
    d_initvalu_32 = d_initvalu[offset_32];
    d_initvalu_33 = d_initvalu[offset_33];
    d_initvalu_34 = d_initvalu[offset_34];
    d_initvalu_35 = d_initvalu[offset_35];
    d_initvalu_36 = d_initvalu[offset_36];
    d_initvalu_37 = d_initvalu[offset_37];
    d_initvalu_38 = d_initvalu[offset_38];
    d_initvalu_39 = d_initvalu[offset_39];
    d_initvalu_40 = d_initvalu[offset_40];
    // d_initvalu_41 = d_initvalu[offset_41];
    // d_initvalu_42 = d_initvalu[offset_42];
    // d_initvalu_43 = d_initvalu[offset_43];
    // d_initvalu_44 = d_initvalu[offset_44];
    // d_initvalu_45 = d_initvalu[offset_45];
    // d_initvalu_46 = d_initvalu[offset_46];

    // matlab constants undefined in c
    pi = (fp)3.1416;

    // Constants
    R    = (fp)8314;             // [J/kmol*K]
    Frdy = (fp)96485;            // [C/mol]
    Temp = (fp)310;              // [K] 310
    FoRT = Frdy / R / Temp;      //
    Cmem = (fp)1.3810e-10;       // [F] membrane capacitance
    Qpow = (Temp - (fp)310) / (fp)10;

    // Cell geometry
    cellLength = (fp)100;     // cell length [um]
    cellRadius = (fp)10.25;   // cell radius [um]
    Vcell      = pi * cellRadius * cellRadius * cellLength * (fp)1e-15; // [L]
    Vmyo       = (fp)0.65  * Vcell;
    Vsr        = (fp)0.035 * Vcell;
    Vsl        = (fp)0.02  * Vcell;
    Vjunc      = (fp)0.0539 * (fp)0.01 * Vcell;
    J_ca_juncsl = (fp)1 / (fp)1.2134e12;            // [L/msec]
    J_ca_slmyo  = (fp)1 / (fp)2.68510e11;           // [L/msec]
    J_na_juncsl = (fp)1 / ((fp)1.6382e12 / (fp)3 * (fp)100); // [L/msec]
    J_na_slmyo  = (fp)1 / ((fp)1.8308e10 / (fp)3 * (fp)100); // [L/msec]

    // Fractional currents in compartments
    Fjunc     = (fp)0.11;
    Fsl       = (fp)1 - Fjunc;
    Fjunc_CaL = (fp)0.9;
    Fsl_CaL   = (fp)1 - Fjunc_CaL;

    // Fixed ion concentrations
    Cli = (fp)15;   // Intracellular Cl  [mM]
    Clo = (fp)150;  // Extracellular Cl  [mM]
    Ko  = (fp)5.4;  // Extracellular K   [mM]
    Nao = (fp)140;  // Extracellular Na  [mM]
    Cao = (fp)1.8;  // Extracellular Ca  [mM]
    Mgi = (fp)1;    // Intracellular Mg  [mM]

    // Nernst Potentials
    const fp invFoRT = (fp)1 / FoRT;
    ena_junc = invFoRT * log(Nao / d_initvalu_32);          // [mV]
    ena_sl   = invFoRT * log(Nao / d_initvalu_33);          // [mV]
    ek       = invFoRT * log(Ko / d_initvalu_35);           // [mV]
    const fp inv2FoRT = (fp)0.5 * invFoRT;
    eca_junc = inv2FoRT * log(Cao / d_initvalu_36);         // [mV]
    eca_sl   = inv2FoRT * log(Cao / d_initvalu_37);         // [mV]
    ecl      = invFoRT * log(Cli / Clo);                    // [mV]

    // Na transport parameters
    GNa     = (fp)16.0;        // [mS/uF]
    GNaB    = (fp)0.297e-3;    // [mS/uF]
    IbarNaK = (fp)1.90719;     // [uA/uF]
    KmNaip  = (fp)11;          // [mM]
    KmKo    = (fp)1.5;         // [mM]

    // K current parameters
    pNaK   = (fp)0.01833;
    GtoSlow= (fp)0.06; // [mS/uF]
    GtoFast= (fp)0.02; // [mS/uF]
    gkp    = (fp)0.001;

    // Cl current parameters
    GClCa  = (fp)0.109625; // [mS/uF]
    GClB   = (fp)9e-3;     // [mS/uF]
    KdClCa = (fp)100e-3;   // [mM]

    // I_Ca parameters
    pNa    = (fp)1.5e-8; // [cm/sec]
    pCa    = (fp)5.4e-4; // [cm/sec]
    pK     = (fp)2.7e-7; // [cm/sec]
    Q10CaL = (fp)1.8;

    // Ca transport parameters
    IbarNCX   = (fp)9.0;      // [uA/uF]
    KmCai     = (fp)3.59e-3;  // [mM]
    KmCao     = (fp)1.3;      // [mM]
    KmNai     = (fp)12.29;    // [mM]
    KmNao     = (fp)87.5;     // [mM]
    ksat      = (fp)0.27;     // [none]
    nu        = (fp)0.35;     // [none]
    Kdact     = (fp)0.256e-3; // [mM]
    Q10NCX    = (fp)1.57;     // [none]
    IbarSLCaP = (fp)0.0673;   // [uA/uF]
    KmPCa     = (fp)0.5e-3;   // [mM]
    GCaB      = (fp)2.513e-4; // [uA/uF]
    Q10SLCaP  = (fp)2.35;     // [none]

    // SR flux parameters
    Q10SRCaP  = (fp)2.6;        // [none]
    Vmax_SRCaP= (fp)2.86e-4;    // [mM/msec]
    Kmf       = (fp)0.246e-3;   // [mM]
    Kmr       = (fp)1.7;        // [mM]
    hillSRCaP = (fp)1.787;      // [mM]
    ks        = (fp)25;         // [1/ms]
    koCa      = (fp)10;         // [mM^-2 1/ms]
    kom       = (fp)0.06;       // [1/ms]
    kiCa      = (fp)0.5;        // [1/mM/ms]
    kim       = (fp)0.005;      // [1/ms]
    ec50SR    = (fp)0.45;       // [mM]

    // Buffering parameters
    Bmax_Naj     = (fp)7.561;        // [mM]
    Bmax_Nasl    = (fp)1.65;         // [mM]
    koff_na      = (fp)1e-3;         // [1/ms]
    kon_na       = (fp)0.1e-3;       // [1/mM/ms]
    Bmax_TnClow  = (fp)70e-3;        // [mM]
    koff_tncl    = (fp)19.6e-3;      // [1/ms]
    kon_tncl     = (fp)32.7;         // [1/mM/ms]
    Bmax_TnChigh = (fp)140e-3;       // [mM]
    koff_tnchca  = (fp)0.032e-3;     // [1/ms]
    kon_tnchca   = (fp)2.37;         // [1/mM/ms]
    koff_tnchmg  = (fp)3.33e-3;      // [1/ms]
    kon_tnchmg   = (fp)3e-3;         // [1/mM/ms]
    Bmax_myosin  = (fp)140e-3;       // [mM]
    koff_myoca   = (fp)0.46e-3;      // [1/ms]
    kon_myoca    = (fp)13.8;         // [1/mM/ms]
    koff_myomg   = (fp)0.057e-3;     // [1/ms]
    kon_myomg    = (fp)0.0157;       // [1/mM/ms]
    Bmax_SR      = (fp)19 * (fp)0.9e-3; // [mM]
    koff_sr      = (fp)60e-3;        // [1/ms]
    kon_sr       = (fp)100;          // [1/mM/ms]
    Bmax_SLlowsl = (fp)37.38e-3 * Vmyo / Vsl;        // [mM]
    Bmax_SLlowj  = (fp)4.62e-3  * Vmyo / Vjunc * (fp)0.1; // [mM]
    koff_sll     = (fp)1300e-3;      // [1/ms]
    kon_sll      = (fp)100;          // [1/mM/ms]
    Bmax_SLhighsl= (fp)13.35e-3 * Vmyo / Vsl;        // [mM]
    Bmax_SLhighj = (fp)1.65e-3  * Vmyo / Vjunc * (fp)0.1; // [mM]
    koff_slh     = (fp)30e-3;        // [1/ms]
    kon_slh      = (fp)100;          // [1/mM/ms]
    Bmax_Csqn    = (fp)2.7;          // [mM]
    koff_csqn    = (fp)65;           // [1/ms]
    kon_csqn     = (fp)100;          // [1/mM/ms]

    // I_Na: Fast Na Current
    const fp v = d_initvalu_39;
    am = (fp)0.32 * (v + (fp)47.13) /
         ((fp)1 - expf((fp)-0.1f * (v + (fp)47.13)));
    bm = (fp)0.08 * expf(-(v / (fp)11));
    if (v >= (fp)-40) {
        ah = (fp)0;
        aj = (fp)0;
        bh = (fp)1 / ((fp)0.13 * ((fp)1 + expf(-(v + (fp)10.66) / (fp)11.1)));
        bj = (fp)0.3f * expf((fp)-2.535e-7f * v) /
             ((fp)1 + expf((fp)-0.1f * (v + (fp)32)));
    } else {
        ah = (fp)0.135f * expf(((fp)80 + v) / (fp)-6.8f);
        bh = (fp)3.56f * expf((fp)0.079f * v) +
             (fp)3.1e5f * expf((fp)0.35f * v);
        aj = (fp)(-127140) * expf((fp)0.2444f * v) -
             (fp)3.474e-5f * expf((fp)-0.04391f * v);
        aj = aj * (v + (fp)37.78f) /
             ((fp)1 + expf((fp)0.311f * (v + (fp)79.23f)));
        bj = (fp)0.1212f * expf((fp)-0.01052f * v) /
             ((fp)1 + expf((fp)-0.1378f * (v + (fp)40.14f)));
    }
    d_finavalu[offset_1] = am * ((fp)1 - d_initvalu_1) - bm * d_initvalu_1;
    d_finavalu[offset_2] = ah * ((fp)1 - d_initvalu_2) - bh * d_initvalu_2;
    d_finavalu[offset_3] = aj * ((fp)1 - d_initvalu_3) - bj * d_initvalu_3;

    const fp m3 = d_initvalu_1 * d_initvalu_1 * d_initvalu_1;
    I_Na_junc = Fjunc * GNa * m3 * d_initvalu_2 *
                d_initvalu_3 * (v - ena_junc);
    I_Na_sl   = Fsl * GNa   * m3 * d_initvalu_2 *
                d_initvalu_3 * (v - ena_sl);

    // I_nabk: Na Background Current
    I_nabk_junc = Fjunc * GNaB * (v - ena_junc);
    I_nabk_sl   = Fsl   * GNaB * (v - ena_sl);

    // I_nak: Na/K Pump Current
    sigma = (expf(Nao / (fp)67.3f) - (fp)1) / (fp)7;
    fnak  = (fp)1 / ((fp)1 + (fp)0.1245f * expf((fp)-0.1f * v * FoRT) +
                     (fp)0.0365f * sigma * expf((fp)-v * FoRT));
    const fp KmNaip4 = KmNaip * KmNaip * KmNaip * KmNaip;
    I_nak_junc = Fjunc * IbarNaK * fnak * Ko /
                 ((fp)1 + powf((KmNaip / d_initvalu_32), (fp)4)) / (Ko + KmKo);
    I_nak_sl   = Fsl   * IbarNaK * fnak * Ko /
                 ((fp)1 + powf((KmNaip / d_initvalu_33), (fp)4)) / (Ko + KmKo);
    I_nak = I_nak_junc + I_nak_sl;

    // I_kr: Rapidly Activating K Current
    gkr   = (fp)0.03f * sqrtf(Ko / (fp)5.4f);
    xrss  = (fp)1 / ((fp)1 + expf((-(v + (fp)50)) / (fp)7.5f));
    tauxr = (fp)1 / ((fp)0.00138f * (v + (fp)7) /
                     ((fp)1 - expf((fp)-0.123f * (v + (fp)7))) +
                     (fp)6.1e-4f * (v + (fp)10) /
                     (expf((fp)0.145f * (v + (fp)10)) - (fp)1));
    d_finavalu[offset_12] = (xrss - d_initvalu_12) / tauxr;
    rkr   = (fp)1 / ((fp)1 + expf((v + (fp)33) / (fp)22.4f));
    I_kr  = gkr * d_initvalu_12 * rkr * (v - ek);

    // I_ks: Slowly Activating K Current
    pcaks_junc = -(fp)log10(d_initvalu_36) + (fp)3.0;
    pcaks_sl   = -(fp)log10(d_initvalu_37) + (fp)3.0;
    gks_junc = (fp)0.07f * ((fp)0.057f +
                 (fp)0.19f / ((fp)1 + expf(((- (fp)7.2f + pcaks_junc) / (fp)0.6f))));
    gks_sl   = (fp)0.07f * ((fp)0.057f +
                 (fp)0.19f / ((fp)1 + expf(((- (fp)7.2f + pcaks_sl) / (fp)0.6f))));
    eks = invFoRT *
          logf((Ko + pNaK * Nao) / (d_initvalu_35 + pNaK * d_initvalu_34));
    xsss  = (fp)1 / ((fp)1 + expf((-(v - (fp)1.5f)) / (fp)16.7f));
    tauxs = (fp)1 / ((fp)7.19e-5f * (v + (fp)30) /
                     ((fp)1 - expf((fp)-0.148f * (v + (fp)30))) +
                     (fp)1.31e-4f * (v + (fp)30) /
                     (expf((fp)0.0687f * (v + (fp)30)) - (fp)1));
    d_finavalu[offset_13] = (xsss - d_initvalu_13) / tauxs;
    const fp xs2_j = d_initvalu_12 * d_initvalu_12;
    const fp xs2_sl= d_initvalu_13 * d_initvalu_13;
    I_ks_junc = Fjunc * gks_junc * xs2_j  * (v - eks);
    I_ks_sl   = Fsl   * gks_sl   * xs2_sl * (v - eks);
    I_ks      = I_ks_junc + I_ks_sl;

    // I_kp: Plateau K current
    kp_kp   = (fp)1 / ((fp)1 + expf((fp)7.488f - v / (fp)5.98f));
    I_kp_junc = Fjunc * gkp * kp_kp * (v - ek);
    I_kp_sl   = Fsl   * gkp * kp_kp * (v - ek);
    I_kp      = I_kp_junc + I_kp_sl;

    // I_to: Transient Outward K Current (slow and fast components)
    xtoss  = (fp)1 / ((fp)1 + expf((-(v + (fp)3.0f)) / (fp)15));
    ytoss  = (fp)1 / ((fp)1 + expf((v + (fp)33.5f) / (fp)10));
    rtoss  = ytoss;
    tauxtos = (fp)9 / ((fp)1 + expf((v + (fp)3.0f) / (fp)15)) + (fp)0.5f;
    tauytos = (fp)3e3f / ((fp)1 + expf((v + (fp)60.0f) / (fp)10)) + (fp)30;
    taurtos = (fp)2800 / ((fp)1 + expf((v + (fp)60.0f) / (fp)10)) + (fp)220;
    d_finavalu[offset_8]  = (xtoss - d_initvalu_8)  / tauxtos;
    d_finavalu[offset_9]  = (ytoss - d_initvalu_9)  / tauytos;
    d_finavalu[offset_40] = (rtoss - d_initvalu_40) / taurtos;
    I_tos = GtoSlow * d_initvalu_8 *
            (d_initvalu_9 + (fp)0.5f * d_initvalu_40) *
            (v - ek); // [uA/uF]

    tauxtof = (fp)3.5f * expf(-v * v / ((fp)30 * (fp)30)) + (fp)1.5f;
    tauytof = (fp)20.0f / ((fp)1 + expf((v + (fp)33.5f) / (fp)10)) + (fp)20.0f;
    d_finavalu[offset_10] = (xtoss - d_initvalu_10) / tauxtof;
    d_finavalu[offset_11] = (ytoss - d_initvalu_11) / tauytof;
    I_tof = GtoFast * d_initvalu_10 * d_initvalu_11 * (v - ek);
    I_to  = I_tos + I_tof;

    // I_ki: Time-Independent K Current
    aki  = (fp)1.02f / ((fp)1 + expf((fp)0.2385f * (v - ek - (fp)59.215f)));
    bki  = ((fp)0.49124f * expf((fp)0.08032f * (v + (fp)5.476f - ek)) +
            expf((fp)0.06175f * (v - ek - (fp)594.31f))) /
           ((fp)1 + expf((fp)-0.5143f * (v - ek + (fp)4.753f)));
    kiss = aki / (aki + bki);
    I_ki = (fp)0.9f * sqrtf(Ko / (fp)5.4f) * kiss * (v - ek);

    // I_ClCa: Ca-activated Cl Current, I_Clbk: background Cl Current
    I_ClCa_junc = Fjunc * GClCa / ((fp)1 + KdClCa / d_initvalu_36) * (v - ecl);
    I_ClCa_sl   = Fsl   * GClCa / ((fp)1 + KdClCa / d_initvalu_37) * (v - ecl);
    I_ClCa      = I_ClCa_junc + I_ClCa_sl;
    I_Clbk      = GClB * (v - ecl);

    // I_Ca: L-type Calcium Current
    dss = (fp)1 / ((fp)1 + expf((-(v + (fp)14.5f)) / (fp)6.0f));
    taud = dss * ((fp)1 - expf((-(v + (fp)14.5f)) / (fp)6.0f)) /
           ((fp)0.035f * (v + (fp)14.5f));
    fss = (fp)1 / ((fp)1 + expf((v + (fp)35.06f) / (fp)3.6f)) +
          (fp)0.6f / ((fp)1 + expf(((fp)50 - v) / (fp)20));
    const fp dv = v + (fp)14.5f;
    const fp t1 = (fp)0.0337f * dv;
    tauf = (fp)1 / ((fp)0.0197f * expf(-(t1 * t1)) + (fp)0.02f);
    d_finavalu[offset_4] = (dss - d_initvalu_4) / taud;
    d_finavalu[offset_5] = (fss - d_initvalu_5) / tauf;
    d_finavalu[offset_6] = (fp)1.7f * d_initvalu_36 * ((fp)1 - d_initvalu_6) -
                           (fp)11.9e-3f * d_initvalu_6; // fCa_junc
    d_finavalu[offset_7] = (fp)1.7f * d_initvalu_37 * ((fp)1 - d_initvalu_7) -
                           (fp)11.9e-3f * d_initvalu_7; // fCa_sl

    const fp vFoRT   = v * Frdy * FoRT;
    const fp exp2vRT = expf((fp)2 * v * FoRT);
    const fp num_j   = (fp)0.341f * d_initvalu_36 * exp2vRT - (fp)0.341f * Cao;
    const fp num_sl  = (fp)0.341f * d_initvalu_37 * exp2vRT - (fp)0.341f * Cao;
    const fp den2    = exp2vRT - (fp)1;

    ibarca_j  = pCa * (fp)4 * vFoRT * num_j  / den2;
    ibarca_sl = pCa * (fp)4 * vFoRT * num_sl / den2;

    const fp expvRT = expf(v * FoRT);
    const fp den1   = expvRT - (fp)1;
    ibark = pK * vFoRT *
            ((fp)0.75f * d_initvalu_35 * expvRT - (fp)0.75f * Ko) / den1;
    ibarna_j  = pNa * vFoRT *
                ((fp)0.75f * d_initvalu_32 * expvRT - (fp)0.75f * Nao) / den1;
    ibarna_sl = pNa * vFoRT *
                ((fp)0.75f * d_initvalu_33 * expvRT - (fp)0.75f * Nao) / den1;

    const fp q10CaL = powf(Q10CaL, Qpow);
    const fp one_fCa_j = (fp)1 - d_initvalu_6;
    const fp one_fCa_sl= (fp)1 - d_initvalu_7;
    I_Ca_junc = (Fjunc_CaL * ibarca_j  * d_initvalu_4 * d_initvalu_5 *
                 one_fCa_j * q10CaL) * (fp)0.45f;
    I_Ca_sl   = (Fsl_CaL   * ibarca_sl * d_initvalu_4 * d_initvalu_5 *
                 one_fCa_sl* q10CaL) * (fp)0.45f;
    I_Ca      = I_Ca_junc + I_Ca_sl;
    d_finavalu[offset_43] = -I_Ca * Cmem / (Vmyo * (fp)2 * Frdy) * (fp)1e3;

    I_CaK = (ibark * d_initvalu_4 * d_initvalu_5 *
             (Fjunc_CaL * one_fCa_j + Fsl_CaL * one_fCa_sl) * q10CaL) *
            (fp)0.45f;
    I_CaNa_junc = (Fjunc_CaL * ibarna_j * d_initvalu_4 * d_initvalu_5 *
                   one_fCa_j * q10CaL) * (fp)0.45f;
    I_CaNa_sl   = (Fsl_CaL   * ibarna_sl * d_initvalu_4 * d_initvalu_5 *
                   one_fCa_sl* q10CaL) * (fp)0.45f;

    // I_ncx: Na/Ca Exchanger flux
    const fp q10NCX = powf(Q10NCX, Qpow);
    Ka_junc = (fp)1 /
              ((fp)1 + powf((Kdact / d_initvalu_36), (fp)3));
    Ka_sl   = (fp)1 /
              ((fp)1 + powf((Kdact / d_initvalu_37), (fp)3));
    const fp Nai3_j = d_initvalu_32 * d_initvalu_32 * d_initvalu_32;
    const fp Nai3_sl= d_initvalu_33 * d_initvalu_33 * d_initvalu_33;
    const fp Nao3   = Nao * Nao * Nao;

    const fp expNuV  = expf(nu * v * FoRT);
    const fp expNu1V = expf((nu - (fp)1) * v * FoRT);

    s1_junc = expNuV  * Nai3_j  * Cao;
    s1_sl   = expNuV  * Nai3_sl * Cao;
    s2_junc = expNu1V * Nao3    * d_initvalu_36;
    s2_sl   = expNu1V * Nao3    * d_initvalu_37;

    const fp KmNai3  = KmNai * KmNai * KmNai;
    const fp KmNao3  = KmNao * KmNao * KmNao;

    s3_junc = (KmCai * Nao3 * ((fp)1 + powf((d_initvalu_32 / KmNai), (fp)3)) +
               KmNao3 * d_initvalu_36 +
               KmNai3 * Cao * ((fp)1 + d_initvalu_36 / KmCai) +
               KmCao * Nai3_j + Nai3_j * Cao +
               Nao3 * d_initvalu_36) *
              ((fp)1 + ksat * expNu1V);
    s3_sl = (KmCai * Nao3 * ((fp)1 + powf((d_initvalu_33 / KmNai), (fp)3)) +
             KmNao3 * d_initvalu_37 +
             KmNai3 * Cao * ((fp)1 + d_initvalu_37 / KmCai) +
             KmCao * Nai3_sl + Nai3_sl * Cao +
             Nao3 * d_initvalu_37) *
            ((fp)1 + ksat * expNu1V);
    I_ncx_junc = Fjunc * IbarNCX * q10NCX * Ka_junc *
                 (s1_junc - s2_junc) / s3_junc;
    I_ncx_sl   = Fsl   * IbarNCX * q10NCX * Ka_sl   *
                 (s1_sl   - s2_sl)   / s3_sl;
    I_ncx      = I_ncx_junc + I_ncx_sl;
    d_finavalu[offset_45] = (fp)2 * I_ncx * Cmem / (Vmyo * (fp)2 * Frdy) * (fp)1e3;

    // I_pca: Sarcolemmal Ca Pump Current
    const fp q10SLCaP = powf(Q10SLCaP, Qpow);
    const fp pow16_KmPCa = powf(KmPCa, (fp)1.6f);
    const fp pow16_Caj   = powf(d_initvalu_36, (fp)1.6f);
    const fp pow16_Casl  = powf(d_initvalu_37, (fp)1.6f);
    I_pca_junc = Fjunc * q10SLCaP * IbarSLCaP *
                 pow16_Caj / (pow16_KmPCa + pow16_Caj);
    I_pca_sl   = Fsl   * q10SLCaP * IbarSLCaP *
                 pow16_Casl / (pow16_KmPCa + pow16_Casl);
    I_pca      = I_pca_junc + I_pca_sl;
    d_finavalu[offset_44] = -I_pca * Cmem / (Vmyo * (fp)2 * Frdy) * (fp)1e3;

    // I_cabk: Ca Background Current
    I_cabk_junc = Fjunc * GCaB * (v - eca_junc);
    I_cabk_sl   = Fsl   * GCaB * (v - eca_sl);
    I_cabk      = I_cabk_junc + I_cabk_sl;
    d_finavalu[offset_46] = -I_cabk * Cmem / (Vmyo * (fp)2 * Frdy) * (fp)1e3;

    // SR fluxes: Calcium Release, SR Ca pump, SR Ca leak
    MaxSR = (fp)15;
    MinSR = (fp)1;
    const fp ec50_over_CaSR = ec50SR / d_initvalu_31;
    kCaSR = MaxSR - (MaxSR - MinSR) /
                    ((fp)1 + powf(ec50_over_CaSR, (fp)2.5f));
    koSRCa = koCa / kCaSR;
    kiSRCa = kiCa * kCaSR;
    RI = (fp)1 - d_initvalu_14 - d_initvalu_15 - d_initvalu_16;
    const fp Ca_junc2 = d_initvalu_36 * d_initvalu_36;
    d_finavalu[offset_14] =
        (kim * RI - kiSRCa * d_initvalu_36 * d_initvalu_14) -
        (koSRCa * Ca_junc2 * d_initvalu_14 - kom * d_initvalu_15); // R
    d_finavalu[offset_15] =
        (koSRCa * Ca_junc2 * d_initvalu_14 - kom * d_initvalu_15) -
        (kiSRCa * d_initvalu_36 * d_initvalu_15 - kim * d_initvalu_16); // O
    d_finavalu[offset_16] =
        (kiSRCa * d_initvalu_36 * d_initvalu_15 - kim * d_initvalu_16) -
        (kom * d_initvalu_16 - koSRCa * Ca_junc2 * RI); // I
    J_SRCarel = ks * d_initvalu_15 *
                (d_initvalu_31 - d_initvalu_36); // [mM/ms]

    const fp q10SRCaP = powf(Q10SRCaP, Qpow);
    const fp Ca_ns   = d_initvalu_38 / Kmf;
    const fp Ca_sr_n = d_initvalu_31 / Kmr;
    const fp Ca_ns_h = powf(Ca_ns, hillSRCaP);
    const fp Ca_sr_h = powf(Ca_sr_n, hillSRCaP);
    J_serca = q10SRCaP * Vmax_SRCaP *
              (Ca_ns_h - Ca_sr_h) /
              ((fp)1 + Ca_ns_h + Ca_sr_h);
    J_SRleak = (fp)5.348e-6f * (d_initvalu_31 - d_initvalu_36); //   [mM/ms]

    // Sodium and Calcium Buffering
    d_finavalu[offset_17] =
        kon_na * d_initvalu_32 * (Bmax_Naj - d_initvalu_17) -
        koff_na * d_initvalu_17; // NaBj      [mM/ms]
    d_finavalu[offset_18] =
        kon_na * d_initvalu_33 * (Bmax_Nasl - d_initvalu_18) -
        koff_na * d_initvalu_18; // NaBsl     [mM/ms]

    // Cytosolic Ca Buffers
    d_finavalu[offset_19] =
        kon_tncl * d_initvalu_38 * (Bmax_TnClow - d_initvalu_19) -
        koff_tncl * d_initvalu_19; // TnCL      [mM/ms]
    d_finavalu[offset_20] =
        kon_tnchca * d_initvalu_38 *
            (Bmax_TnChigh - d_initvalu_20 - d_initvalu_21) -
        koff_tnchca * d_initvalu_20; // TnCHc     [mM/ms]
    d_finavalu[offset_21] =
        kon_tnchmg * Mgi *
            (Bmax_TnChigh - d_initvalu_20 - d_initvalu_21) -
        koff_tnchmg * d_initvalu_21; // TnCHm     [mM/ms]
    d_finavalu[offset_22] = (fp)0;  // CaM       [mM/ms]
    d_finavalu[offset_23] =
        kon_myoca * d_initvalu_38 *
            (Bmax_myosin - d_initvalu_23 - d_initvalu_24) -
        koff_myoca * d_initvalu_23; // Myosin_ca [mM/ms]
    d_finavalu[offset_24] =
        kon_myomg * Mgi *
            (Bmax_myosin - d_initvalu_23 - d_initvalu_24) -
        koff_myomg * d_initvalu_24; // Myosin_mg [mM/ms]
    d_finavalu[offset_25] =
        kon_sr * d_initvalu_38 * (Bmax_SR - d_initvalu_25) -
        koff_sr * d_initvalu_25; // SRB       [mM/ms]
    J_CaB_cytosol = d_finavalu[offset_19] + d_finavalu[offset_20] +
                    d_finavalu[offset_21] + d_finavalu[offset_22] +
                    d_finavalu[offset_23] + d_finavalu[offset_24] +
                    d_finavalu[offset_25];

    // Junctional and SL Ca Buffers
    d_finavalu[offset_26] =
        kon_sll * d_initvalu_36 * (Bmax_SLlowj - d_initvalu_26) -
        koff_sll * d_initvalu_26; // SLLj      [mM/ms]
    d_finavalu[offset_27] =
        kon_sll * d_initvalu_37 * (Bmax_SLlowsl - d_initvalu_27) -
        koff_sll * d_initvalu_27; // SLLsl     [mM/ms]
    d_finavalu[offset_28] =
        kon_slh * d_initvalu_36 * (Bmax_SLhighj - d_initvalu_28) -
        koff_slh * d_initvalu_28; // SLHj      [mM/ms]
    d_finavalu[offset_29] =
        kon_slh * d_initvalu_37 * (Bmax_SLhighsl - d_initvalu_29) -
        koff_slh * d_initvalu_29; // SLHsl     [mM/ms]
    J_CaB_junction = d_finavalu[offset_26] + d_finavalu[offset_28];
    J_CaB_sl       = d_finavalu[offset_27] + d_finavalu[offset_29];

    // SR Ca Concentrations
    d_finavalu[offset_30] =
        kon_csqn * d_initvalu_31 * (Bmax_Csqn - d_initvalu_30) -
        koff_csqn * d_initvalu_30; // Csqn      [mM/ms]
    oneovervsr = (fp)1 / Vsr;
    d_finavalu[offset_31] =
        J_serca * Vmyo * oneovervsr -
        (J_SRleak * Vmyo * oneovervsr + J_SRCarel) -
        d_finavalu[offset_30]; // Ca_sr     [mM/ms]

    // Sodium Concentrations
    I_Na_tot_junc = I_Na_junc + I_nabk_junc + (fp)3 * I_ncx_junc +
                    (fp)3 * I_nak_junc + I_CaNa_junc; // [uA/uF]
    I_Na_tot_sl = I_Na_sl + I_nabk_sl + (fp)3 * I_ncx_sl +
                  (fp)3 * I_nak_sl + I_CaNa_sl; // [uA/uF]
    d_finavalu[offset_32] =
        -I_Na_tot_junc * Cmem / (Vjunc * Frdy) +
        J_na_juncsl / Vjunc * (d_initvalu_33 - d_initvalu_32) -
        d_finavalu[offset_17];
    oneovervsl = (fp)1 / Vsl;
    d_finavalu[offset_33] =
        -I_Na_tot_sl * Cmem * oneovervsl / Frdy +
        J_na_juncsl * oneovervsl * (d_initvalu_32 - d_initvalu_33) +
        J_na_slmyo * oneovervsl * (d_initvalu_34 - d_initvalu_33) -
        d_finavalu[offset_18];
    d_finavalu[offset_34] =
        J_na_slmyo / Vmyo * (d_initvalu_33 - d_initvalu_34); // [mM/msec]

    // Potassium Concentration
    I_K_tot = I_to + I_kr + I_ks + I_ki - (fp)2 * I_nak + I_CaK + I_kp; // [uA/uF]
    d_finavalu[offset_35] = (fp)0;                                      // [mM/msec]

    // Calcium Concentrations
    I_Ca_tot_junc =
        I_Ca_junc + I_cabk_junc + I_pca_junc - (fp)2 * I_ncx_junc;   // [uA/uF]
    I_Ca_tot_sl = I_Ca_sl + I_cabk_sl + I_pca_sl -
                  (fp)2 * I_ncx_sl; // [uA/uF]
    d_finavalu[offset_36] =
        -I_Ca_tot_junc * Cmem / (Vjunc * (fp)2 * Frdy) +
        J_ca_juncsl / Vjunc * (d_initvalu_37 - d_initvalu_36) -
        J_CaB_junction +
        J_SRCarel * Vsr / Vjunc + J_SRleak * Vmyo / Vjunc; // Ca_j
    d_finavalu[offset_37] =
        -I_Ca_tot_sl * Cmem / (Vsl * (fp)2 * Frdy) +
        J_ca_juncsl / Vsl * (d_initvalu_36 - d_initvalu_37) +
        J_ca_slmyo / Vsl * (d_initvalu_38 - d_initvalu_37) -
        J_CaB_sl; // Ca_sl
    d_finavalu[offset_38] = -J_serca - J_CaB_cytosol +
                            J_ca_slmyo / Vmyo *
                                (d_initvalu_37 - d_initvalu_38);

    // Simulation type
    state = 1;
    switch (state) {
    case 0:
        I_app = (fp)0;
        break;
    case 1: // pace w/ current injection at cycleLength 'cycleLength'
        if (fmod(timeinst, cycleLength) <= (fp)5) {
            I_app = (fp)9.5;
        } else {
            I_app = (fp)0.0;
        }
        break;
    case 2:
        V_hold = (fp)-55;
        V_test = (fp)0;
        if (timeinst > (fp)0.5f & timeinst < (fp)200.5f) {
            V_clamp = V_test;
        } else {
            V_clamp = V_hold;
        }
        R_clamp = (fp)0.04;
        I_app = (V_clamp - v) / R_clamp;
        break;
    }

    // Membrane Potential
    I_Na_tot = I_Na_tot_junc + I_Na_tot_sl; // [uA/uF]
    I_Cl_tot = I_ClCa + I_Clbk;             // [uA/uF]
    I_Ca_tot = I_Ca_tot_junc + I_Ca_tot_sl;
    I_tot    = I_Na_tot + I_Cl_tot + I_Ca_tot + I_K_tot;
    d_finavalu[offset_39] = -(I_tot - I_app);

    // Set unused output values to 0 (MATLAB does it by default)
    d_finavalu[offset_41] = (fp)0;
    d_finavalu[offset_42] = (fp)0;
}
