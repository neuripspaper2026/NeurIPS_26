__device__ void kernel_ecc(fp timeinst, fp * __restrict__ d_initvalu,
                           fp * __restrict__ d_finavalu,
                           const int valu_offset,
                           fp * __restrict__ d_params) {

    //=====================================================================
    //	VARIABLES
    //=====================================================================

    // input parameters
    const fp cycleLength = d_params[15];

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
    const int offset_16 = valu_offset + 15;
    const int offset_17 = valu_offset + 16;
    const int offset_18 = valu_offset + 17;
    const int offset_19 = valu_offset + 18;
    const int offset_20 = valu_offset + 19;
    const int offset_21 = valu_offset + 20;
    const int offset_22 = valu_offset + 21;
    const int offset_23 = valu_offset + 22;
    const int offset_24 = valu_offset + 23;
    const int offset_25 = valu_offset + 24;
    const int offset_26 = valu_offset + 25;
    const int offset_27 = valu_offset + 26;
    const int offset_28 = valu_offset + 27;
    const int offset_29 = valu_offset + 28;
    const int offset_30 = valu_offset + 29;
    const int offset_31 = valu_offset + 30;
    const int offset_32 = valu_offset + 31;
    const int offset_33 = valu_offset + 32;
    const int offset_34 = valu_offset + 33;
    const int offset_35 = valu_offset + 34;
    const int offset_36 = valu_offset + 35;
    const int offset_37 = valu_offset + 36;
    const int offset_38 = valu_offset + 37;
    const int offset_39 = valu_offset + 38;
    const int offset_40 = valu_offset + 39;
    const int offset_41 = valu_offset + 40;
    const int offset_42 = valu_offset + 41;
    const int offset_43 = valu_offset + 42;
    const int offset_44 = valu_offset + 43;
    const int offset_45 = valu_offset + 44;
    const int offset_46 = valu_offset + 45;

    // stored input array (use registers)
    fp d_initvalu_1  = d_initvalu[offset_1];
    fp d_initvalu_2  = d_initvalu[offset_2];
    fp d_initvalu_3  = d_initvalu[offset_3];
    fp d_initvalu_4  = d_initvalu[offset_4];
    fp d_initvalu_5  = d_initvalu[offset_5];
    fp d_initvalu_6  = d_initvalu[offset_6];
    fp d_initvalu_7  = d_initvalu[offset_7];
    fp d_initvalu_8  = d_initvalu[offset_8];
    fp d_initvalu_9  = d_initvalu[offset_9];
    fp d_initvalu_10 = d_initvalu[offset_10];
    fp d_initvalu_11 = d_initvalu[offset_11];
    fp d_initvalu_12 = d_initvalu[offset_12];
    fp d_initvalu_13 = d_initvalu[offset_13];
    fp d_initvalu_14 = d_initvalu[offset_14];
    fp d_initvalu_15 = d_initvalu[offset_15];
    fp d_initvalu_16 = d_initvalu[offset_16];
    fp d_initvalu_17 = d_initvalu[offset_17];
    fp d_initvalu_18 = d_initvalu[offset_18];
    fp d_initvalu_19 = d_initvalu[offset_19];
    fp d_initvalu_20 = d_initvalu[offset_20];
    fp d_initvalu_21 = d_initvalu[offset_21];
    fp d_initvalu_23 = d_initvalu[offset_23];
    fp d_initvalu_24 = d_initvalu[offset_24];
    fp d_initvalu_25 = d_initvalu[offset_25];
    fp d_initvalu_26 = d_initvalu[offset_26];
    fp d_initvalu_27 = d_initvalu[offset_27];
    fp d_initvalu_28 = d_initvalu[offset_28];
    fp d_initvalu_29 = d_initvalu[offset_29];
    fp d_initvalu_30 = d_initvalu[offset_30];
    fp d_initvalu_31 = d_initvalu[offset_31];
    fp d_initvalu_32 = d_initvalu[offset_32];
    fp d_initvalu_33 = d_initvalu[offset_33];
    fp d_initvalu_34 = d_initvalu[offset_34];
    fp d_initvalu_35 = d_initvalu[offset_35];
    fp d_initvalu_36 = d_initvalu[offset_36];
    fp d_initvalu_37 = d_initvalu[offset_37];
    fp d_initvalu_38 = d_initvalu[offset_38];
    fp d_initvalu_39 = d_initvalu[offset_39];
    fp d_initvalu_40 = d_initvalu[offset_40];

    // matlab constants undefined in c
    const fp pi = (fp)3.1416;

    // Constants
    const fp R    = (fp)8314;        // [J/kmol*K]
    const fp Frdy = (fp)96485;       // [C/mol]
    const fp Temp = (fp)310;         // [K] 310
    const fp FoRT = Frdy / (R * Temp);
    const fp Cmem = (fp)1.3810e-10;  // [F] membrane capacitance
    const fp Qpow = (Temp - (fp)310) / (fp)10;

    // Cell geometry
    const fp cellLength = (fp)100;     // cell length [um]
    const fp cellRadius = (fp)10.25;   // cell radius [um]
    const fp Vcell      = pi * cellRadius * cellRadius * cellLength * (fp)1e-15; // [L]
    const fp Vmyo       = (fp)0.65  * Vcell;
    const fp Vsr        = (fp)0.035 * Vcell;
    const fp Vsl        = (fp)0.02  * Vcell;
    const fp Vjunc      = (fp)0.0539 * (fp)0.01 * Vcell;
    const fp J_ca_juncsl = (fp)1.0 / (fp)1.2134e12;              // [L/msec]
    const fp J_ca_slmyo  = (fp)1.0 / (fp)2.68510e11;             // [L/msec]
    const fp J_na_juncsl = (fp)1.0 / ((fp)1.6382e12 / (fp)3 * (fp)100); // [L/msec]
    const fp J_na_slmyo  = (fp)1.0 / ((fp)1.8308e10 / (fp)3 * (fp)100);  // [L/msec]

    // Fractional currents in compartments
    const fp Fjunc     = (fp)0.11;
    const fp Fsl       = (fp)1.0 - Fjunc;
    const fp Fjunc_CaL = (fp)0.9;
    const fp Fsl_CaL   = (fp)1.0 - Fjunc_CaL;

    // Fixed ion concentrations
    const fp Cli = (fp)15;   // [mM]
    const fp Clo = (fp)150;  // [mM]
    const fp Ko  = (fp)5.4;  // [mM]
    const fp Nao = (fp)140;  // [mM]
    const fp Cao = (fp)1.8;  // [mM]
    const fp Mgi = (fp)1.0;  // [mM]

    // Nernst Potentials
    const fp invFoRT = (fp)1.0 / FoRT;
    const fp ena_junc = invFoRT * log(Nao / d_initvalu_32);           // [mV]
    const fp ena_sl   = invFoRT * log(Nao / d_initvalu_33);           // [mV]
    const fp ek       = invFoRT * log(Ko / d_initvalu_35);            // [mV]
    const fp eca_junc = invFoRT * (fp)0.5 * log(Cao / d_initvalu_36); // [mV]
    const fp eca_sl   = invFoRT * (fp)0.5 * log(Cao / d_initvalu_37); // [mV]
    const fp ecl      = invFoRT * log(Cli / Clo);                     // [mV]

    // Na transport parameters
    const fp GNa     = (fp)16.0;       // [mS/uF]
    const fp GNaB    = (fp)0.297e-3;   // [mS/uF]
    const fp IbarNaK = (fp)1.90719;    // [uA/uF]
    const fp KmNaip  = (fp)11.0;       // [mM]
    const fp KmKo    = (fp)1.5;        // [mM]

    // K current parameters
    const fp pNaK    = (fp)0.01833;
    const fp GtoSlow = (fp)0.06;  // [mS/uF]
    const fp GtoFast = (fp)0.02;  // [mS/uF]
    const fp gkp     = (fp)0.001;

    // Cl current parameters
    const fp GClCa  = (fp)0.109625; // [mS/uF]
    const fp GClB   = (fp)9e-3;     // [mS/uF]
    const fp KdClCa = (fp)100e-3;   // [mM]

    // I_Ca parameters
    const fp pNa     = (fp)1.5e-8;  // [cm/sec]
    const fp pCa     = (fp)5.4e-4;  // [cm/sec]
    const fp pK      = (fp)2.7e-7;  // [cm/sec]
    const fp Q10CaL  = (fp)1.8;

    // Ca transport parameters
    const fp IbarNCX   = (fp)9.0;        // [uA/uF]
    const fp KmCai     = (fp)3.59e-3;    // [mM]
    const fp KmCao     = (fp)1.3;        // [mM]
    const fp KmNai     = (fp)12.29;      // [mM]
    const fp KmNao     = (fp)87.5;       // [mM]
    const fp ksat      = (fp)0.27;       // [none]
    const fp nu        = (fp)0.35;       // [none]
    const fp Kdact     = (fp)0.256e-3;   // [mM]
    const fp Q10NCX    = (fp)1.57;       // [none]
    const fp IbarSLCaP = (fp)0.0673;     // [uA/uF]
    const fp KmPCa     = (fp)0.5e-3;     // [mM]
    const fp GCaB      = (fp)2.513e-4;   // [uA/uF]
    const fp Q10SLCaP  = (fp)2.35;       // [none]

    // SR flux parameters
    const fp Q10SRCaP   = (fp)2.6;        // [none]
    const fp Vmax_SRCaP = (fp)2.86e-4;    // [mM/msec]
    const fp Kmf        = (fp)0.246e-3;   // [mM]
    const fp Kmr        = (fp)1.7;        // [mM]
    const fp hillSRCaP  = (fp)1.787;      // [mM]
    const fp ks         = (fp)25.0;       // [1/ms]
    const fp koCa       = (fp)10.0;       // [mM^-2 1/ms]
    const fp kom        = (fp)0.06;       // [1/ms]
    const fp kiCa       = (fp)0.5;        // [1/mM/ms]
    const fp kim        = (fp)0.005;      // [1/ms]
    const fp ec50SR     = (fp)0.45;       // [mM]

    // Buffering parameters
    const fp Bmax_Naj      = (fp)7.561;        // [mM]
    const fp Bmax_Nasl     = (fp)1.65;        // [mM]
    const fp koff_na       = (fp)1e-3;        // [1/ms]
    const fp kon_na        = (fp)0.1e-3;      // [1/mM/ms]
    const fp Bmax_TnClow   = (fp)70e-3;       // [mM]
    const fp koff_tncl     = (fp)19.6e-3;     // [1/ms]
    const fp kon_tncl      = (fp)32.7;        // [1/mM/ms]
    const fp Bmax_TnChigh  = (fp)140e-3;      // [mM]
    const fp koff_tnchca   = (fp)0.032e-3;    // [1/ms]
    const fp kon_tnchca    = (fp)2.37;        // [1/mM/ms]
    const fp koff_tnchmg   = (fp)3.33e-3;     // [1/ms]
    const fp kon_tnchmg    = (fp)3e-3;        // [1/mM/ms]
    const fp Bmax_myosin   = (fp)140e-3;      // [mM]
    const fp koff_myoca    = (fp)0.46e-3;     // [1/ms]
    const fp kon_myoca     = (fp)13.8;        // [1/mM/ms]
    const fp koff_myomg    = (fp)0.057e-3;    // [1/ms]
    const fp kon_myomg     = (fp)0.0157;      // [1/mM/ms]
    const fp Bmax_SR       = (fp)(19.0 * 0.9e-3); // [mM]
    const fp koff_sr       = (fp)60e-3;       // [1/ms]
    const fp kon_sr        = (fp)100.0;       // [1/mM/ms]
    const fp Bmax_SLlowsl  = (fp)(37.38e-3) * Vmyo / Vsl;
    const fp Bmax_SLlowj   = (fp)(4.62e-3) * Vmyo / Vjunc * (fp)0.1;
    const fp koff_sll      = (fp)1300e-3;     // [1/ms]
    const fp kon_sll       = (fp)100.0;       // [1/mM/ms]
    const fp Bmax_SLhighsl = (fp)(13.35e-3) * Vmyo / Vsl;
    const fp Bmax_SLhighj  = (fp)(1.65e-3) * Vmyo / Vjunc * (fp)0.1;
    const fp koff_slh      = (fp)30e-3;       // [1/ms]
    const fp kon_slh       = (fp)100.0;       // [1/mM/ms]
    const fp Bmax_Csqn     = (fp)2.7;         // [mM]
    const fp koff_csqn     = (fp)65.0;        // [1/ms]
    const fp kon_csqn      = (fp)100.0;       // [1/mM/ms]

    // I_Na: Fast Na Current
    fp am, bm, ah, bh, aj, bj;
    fp I_Na_junc, I_Na_sl;

    // I_nabk: Na Background Current
    fp I_nabk_junc, I_nabk_sl;

    // I_nak: Na/K Pump Current
    fp sigma, fnak, I_nak_junc, I_nak_sl, I_nak;

    // I_kr: Rapidly Activating K Current
    fp gkr, xrss, tauxr, rkr, I_kr;

    // I_ks: Slowly Activating K Current
    fp pcaks_junc, pcaks_sl, gks_junc, gks_sl, eks, xsss, tauxs;
    fp I_ks_junc, I_ks_sl, I_ks;

    // I_kp: Plateau K current
    fp kp_kp, I_kp_junc, I_kp_sl, I_kp;

    // I_to: Transient Outward K Current
    fp xtoss, ytoss, rtoss, tauxtos, tauytos, taurtos;
    fp tauxtof, tauytof, I_tos, I_tof, I_to;

    // I_ki
    fp aki, bki, kiss, I_ki;

    // I_ClCa, I_Clbk
    fp I_ClCa_junc, I_ClCa_sl, I_ClCa, I_Clbk;

    // I_Ca
    fp dss, taud, fss, tauf;
    fp ibarca_j, ibarca_sl, ibark, ibarna_j, ibarna_sl;
    fp I_Ca_junc, I_Ca_sl, I_Ca, I_CaK, I_CaNa_junc, I_CaNa_sl;

    // I_ncx
    fp Ka_junc, Ka_sl;
    fp s1_junc, s1_sl, s2_junc, s3_junc, s2_sl, s3_sl;
    fp I_ncx_junc, I_ncx_sl, I_ncx;

    // I_pca
    fp I_pca_junc, I_pca_sl, I_pca;

    // I_cabk
    fp I_cabk_junc, I_cabk_sl, I_cabk;

    // SR fluxes
    fp MaxSR, MinSR, kCaSR, koSRCa, kiSRCa, RI;
    fp J_SRCarel, J_serca, J_SRleak;

    // Cytosolic Ca Buffers
    fp J_CaB_cytosol;

    // Junctional and SL Ca Buffers
    fp J_CaB_junction, J_CaB_sl;

    // SR Ca Concentrations
    fp oneovervsr;

    // Sodium Concentrations
    fp I_Na_tot_junc, I_Na_tot_sl, oneovervsl;

    // Potassium Concentration
    fp I_K_tot;

    // Calcium Concentrations
    fp I_Ca_tot_junc, I_Ca_tot_sl;

    // Simulation type
    int state;
    fp I_app = (fp)0.0;
    fp V_hold, V_test, V_clamp, R_clamp;

    // Membrane Potential
    fp I_Na_tot, I_Cl_tot, I_Ca_tot, I_tot;

    //=====================================================================
    //	EXECUTION
    //=====================================================================

    // I_Na: Fast Na Current
    const fp v39 = d_initvalu_39;
    am = (fp)0.32 * (v39 + (fp)47.13) /
         ((fp)1.0 - exp((fp)-0.1 * (v39 + (fp)47.13)));
    bm = (fp)0.08 * exp(-v39 / (fp)11.0);

    if (v39 >= (fp)-40.0) {
        ah = (fp)0.0;
        aj = (fp)0.0;
        bh = (fp)1.0 /
             ((fp)0.13 * ((fp)1.0 + exp(-(v39 + (fp)10.66) / (fp)11.1)));
        bj = (fp)0.3 * exp((fp)-2.535e-7 * v39) /
             ((fp)1.0 + exp((fp)-0.1 * (v39 + (fp)32.0)));
    } else {
        ah = (fp)0.135 * exp(((fp)80.0 + v39) / (fp)-6.8);
        bh = (fp)3.56 * exp((fp)0.079 * v39) +
             (fp)3.1e5 * exp((fp)0.35 * v39);
        aj = ((fp)-127140.0 * exp((fp)0.2444 * v39) -
              (fp)3.474e-5 * exp((fp)-0.04391 * v39)) *
             (v39 + (fp)37.78) /
             ((fp)1.0 + exp((fp)0.311 * (v39 + (fp)79.23)));
        bj = (fp)0.1212 * exp((fp)-0.01052 * v39) /
             ((fp)1.0 + exp((fp)-0.1378 * (v39 + (fp)40.14)));
    }

    d_finavalu[offset_1] = am * ((fp)1.0 - d_initvalu_1) - bm * d_initvalu_1;
    d_finavalu[offset_2] = ah * ((fp)1.0 - d_initvalu_2) - bh * d_initvalu_2;
    d_finavalu[offset_3] = aj * ((fp)1.0 - d_initvalu_3) - bj * d_initvalu_3;

    const fp m3 = d_initvalu_1 * d_initvalu_1 * d_initvalu_1;
    const fp common_na = GNa * m3 * d_initvalu_2 * d_initvalu_3;
    I_Na_junc = Fjunc * common_na * (v39 - ena_junc);
    I_Na_sl   = Fsl   * common_na * (v39 - ena_sl);

    // I_nabk: Na Background Current
    const fp diff_ena_junc = v39 - ena_junc;
    const fp diff_ena_sl   = v39 - ena_sl;
    I_nabk_junc = Fjunc * GNaB * diff_ena_junc;
    I_nabk_sl   = Fsl   * GNaB * diff_ena_sl;

    // I_nak: Na/K Pump Current
    sigma = (exp(Nao / (fp)67.3) - (fp)1.0) / (fp)7.0;
    fnak = (fp)1.0 /
           ((fp)1.0 +
            (fp)0.1245 * exp((fp)-0.1 * v39 * FoRT) +
            (fp)0.0365 * sigma * exp(-v39 * FoRT));
    const fp Ko_plus_KmKo = Ko + KmKo;
    const fp KmNaip_4 = KmNaip * KmNaip * KmNaip * KmNaip;
    I_nak_junc = Fjunc * IbarNaK * fnak * Ko /
                 ((fp)1.0 + pow((KmNaip / d_initvalu_32), (fp)4.0)) /
                 Ko_plus_KmKo;
    I_nak_sl   = Fsl   * IbarNaK * fnak * Ko /
                 ((fp)1.0 + pow((KmNaip / d_initvalu_33), (fp)4.0)) /
                 Ko_plus_KmKo;
    I_nak = I_nak_junc + I_nak_sl;

    // I_kr: Rapidly Activating K Current
    gkr  = (fp)0.03 * sqrt(Ko / (fp)5.4);
    xrss = (fp)1.0 / ((fp)1.0 + exp(-(v39 + (fp)50.0) / (fp)7.5));
    tauxr = (fp)1.0 /
            ((fp)0.00138 * (v39 + (fp)7.0) /
                 ((fp)1.0 - exp((fp)-0.123 * (v39 + (fp)7.0))) +
             (fp)6.1e-4 * (v39 + (fp)10.0) /
                 (exp((fp)0.145 * (v39 + (fp)10.0)) - (fp)1.0));
    d_finavalu[offset_12] = (xrss - d_initvalu_12) / tauxr;
    rkr  = (fp)1.0 / ((fp)1.0 + exp((v39 + (fp)33.0) / (fp)22.4));
    I_kr = gkr * d_initvalu_12 * rkr * (v39 - ek);

    // I_ks: Slowly Activating K Current
    pcaks_junc = -log10(d_initvalu_36) + (fp)3.0;
    pcaks_sl   = -log10(d_initvalu_37) + (fp)3.0;
    gks_junc = (fp)0.07 *
               ((fp)0.057 +
                (fp)0.19 / ((fp)1.0 + exp((-(fp)7.2 + pcaks_junc) / (fp)0.6)));
    gks_sl = (fp)0.07 *
             ((fp)0.057 +
              (fp)0.19 / ((fp)1.0 + exp((-(fp)7.2 + pcaks_sl) / (fp)0.6)));
    eks = invFoRT *
          log((Ko + pNaK * Nao) /
              (d_initvalu_35 + pNaK * d_initvalu_34));
    xsss = (fp)1.0 / ((fp)1.0 + exp(-(v39 - (fp)1.5) / (fp)16.7));
    tauxs = (fp)1.0 /
            ((fp)7.19e-5 * (v39 + (fp)30.0) /
                 ((fp)1.0 - exp((fp)-0.148 * (v39 + (fp)30.0))) +
             (fp)1.31e-4 * (v39 + (fp)30.0) /
                 (exp((fp)0.0687 * (v39 + (fp)30.0)) - (fp)1.0));
    d_finavalu[offset_13] = (xsss - d_initvalu_13) / tauxs;
    const fp ks_gate_j = d_initvalu_12 * d_initvalu_12;
    const fp ks_gate_sl = d_initvalu_13 * d_initvalu_13;
    I_ks_junc = Fjunc * gks_junc * ks_gate_j * (v39 - eks);
    I_ks_sl   = Fsl   * gks_sl   * ks_gate_sl * (v39 - eks);
    I_ks = I_ks_junc + I_ks_sl;

    // I_kp: Plateau K current
    kp_kp = (fp)1.0 / ((fp)1.0 + exp((fp)7.488 - v39 / (fp)5.98));
    const fp diff_vk = v39 - ek;
    I_kp_junc = Fjunc * gkp * kp_kp * diff_vk;
    I_kp_sl   = Fsl   * gkp * kp_kp * diff_vk;
    I_kp = I_kp_junc + I_kp_sl;

    // I_to: Transient Outward K Current (slow and fast components)
    xtoss = (fp)1.0 / ((fp)1.0 + exp(-(v39 + (fp)3.0) / (fp)15.0));
    ytoss = (fp)1.0 / ((fp)1.0 + exp((v39 + (fp)33.5) / (fp)10.0));
    rtoss = (fp)1.0 / ((fp)1.0 + exp((v39 + (fp)33.5) / (fp)10.0));
    tauxtos = (fp)9.0 / ((fp)1.0 + exp((v39 + (fp)3.0) / (fp)15.0)) +
              (fp)0.5;
    tauytos = (fp)3e3 / ((fp)1.0 + exp((v39 + (fp)60.0) / (fp)10.0)) +
              (fp)30.0;
    taurtos = (fp)2800.0 /
                  ((fp)1.0 + exp((v39 + (fp)60.0) / (fp)10.0)) +
              (fp)220.0;
    d_finavalu[offset_8]  = (xtoss - d_initvalu_8) / tauxtos;
    d_finavalu[offset_9]  = (ytoss - d_initvalu_9) / tauytos;
    d_finavalu[offset_40] = (rtoss - d_initvalu_40) / taurtos;
    I_tos = GtoSlow * d_initvalu_8 *
            (d_initvalu_9 + (fp)0.5 * d_initvalu_40) *
            diff_vk;

    tauxtof = (fp)3.5 * exp(-v39 * v39 / ((fp)30.0 * (fp)30.0)) +
              (fp)1.5;
    tauytof = (fp)20.0 /
                  ((fp)1.0 + exp((v39 + (fp)33.5) / (fp)10.0)) +
              (fp)20.0;
    d_finavalu[offset_10] = (xtoss - d_initvalu_10) / tauxtof;
    d_finavalu[offset_11] = (ytoss - d_initvalu_11) / tauytof;
    I_tof = GtoFast * d_initvalu_10 * d_initvalu_11 * diff_vk;
    I_to  = I_tos + I_tof;

    // I_ki: Time-Independent K Current
    aki = (fp)1.02 /
          ((fp)1.0 + exp((fp)0.2385 * (v39 - ek - (fp)59.215)));
    bki = ((fp)0.49124 * exp((fp)0.08032 * (v39 + (fp)5.476 - ek)) +
           exp((fp)0.06175 * (v39 - ek - (fp)594.31))) /
          ((fp)1.0 + exp((fp)-0.5143 * (v39 - ek + (fp)4.753)));
    kiss = aki / (aki + bki);
    I_ki = (fp)0.9 * sqrt(Ko / (fp)5.4) * kiss * diff_vk;

    // I_ClCa: Ca-activated Cl Current, I_Clbk: background Cl Current
    const fp denom_j = (fp)1.0 + KdClCa / d_initvalu_36;
    const fp denom_sl = (fp)1.0 + KdClCa / d_initvalu_37;
    I_ClCa_junc = Fjunc * GClCa / denom_j * (v39 - ecl);
    I_ClCa_sl   = Fsl   * GClCa / denom_sl * (v39 - ecl);
    I_ClCa      = I_ClCa_junc + I_ClCa_sl;
    I_Clbk      = GClB * (v39 - ecl);

    // I_Ca: L-type Calcium Current
    dss = (fp)1.0 /
          ((fp)1.0 + exp(-(v39 + (fp)14.5) / (fp)6.0));
    taud = dss *
           ((fp)1.0 - exp(-(v39 + (fp)14.5) / (fp)6.0)) /
           ((fp)0.035 * (v39 + (fp)14.5));
    fss = (fp)1.0 / ((fp)1.0 + exp((v39 + (fp)35.06) / (fp)3.6)) +
          (fp)0.6 / ((fp)1.0 + exp(((fp)50.0 - v39) / (fp)20.0));
    const fp tmp = (fp)0.0337 * (v39 + (fp)14.5);
    tauf = (fp)1.0 /
           ((fp)0.0197 * exp(-tmp * tmp) + (fp)0.02);
    d_finavalu[offset_4] = (dss - d_initvalu_4) / taud;
    d_finavalu[offset_5] = (fss - d_initvalu_5) / tauf;
    d_finavalu[offset_6] = (fp)1.7 * d_initvalu_36 * ((fp)1.0 - d_initvalu_6) -
                           (fp)11.9e-3 * d_initvalu_6; // fCa_junc
    d_finavalu[offset_7] = (fp)1.7 * d_initvalu_37 * ((fp)1.0 - d_initvalu_7) -
                           (fp)11.9e-3 * d_initvalu_7; // fCa_sl

    // precompute exponentials in Ca currents
    const fp v39FoRT = v39 * FoRT;
    const fp exp_v39FoRT  = exp(v39FoRT);
    const fp exp_2v39FoRT = exp_v39FoRT * exp_v39FoRT;

    ibarca_j =
        pCa * (fp)4.0 * (v39 * Frdy * FoRT) *
        ((fp)0.341 * d_initvalu_36 * exp_2v39FoRT - (fp)0.341 * Cao) /
        (exp_2v39FoRT - (fp)1.0);
    ibarca_sl =
        pCa * (fp)4.0 * (v39 * Frdy * FoRT) *
        ((fp)0.341 * d_initvalu_37 * exp_2v39FoRT - (fp)0.341 * Cao) /
        (exp_2v39FoRT - (fp)1.0);
    ibark =
        pK * (v39 * Frdy * FoRT) *
        ((fp)0.75 * d_initvalu_35 * exp_v39FoRT - (fp)0.75 * Ko) /
        (exp_v39FoRT - (fp)1.0);
    ibarna_j =
        pNa * (v39 * Frdy * FoRT) *
        ((fp)0.75 * d_initvalu_32 * exp_v39FoRT - (fp)0.75 * Nao) /
        (exp_v39FoRT - (fp)1.0);
    ibarna_sl =
        pNa * (v39 * Frdy * FoRT) *
        ((fp)0.75 * d_initvalu_33 * exp_v39FoRT - (fp)0.75 * Nao) /
        (exp_v39FoRT - (fp)1.0);

    const fp gate_common =
        d_initvalu_4 * d_initvalu_5 * pow(Q10CaL, Qpow) * (fp)0.45;
    const fp one_minus_fCa_j = (fp)1.0 - d_initvalu_6;
    const fp one_minus_fCa_sl = (fp)1.0 - d_initvalu_7;

    I_Ca_junc = Fjunc_CaL * ibarca_j * gate_common * one_minus_fCa_j;
    I_Ca_sl   = Fsl_CaL   * ibarca_sl * gate_common * one_minus_fCa_sl;
    I_Ca      = I_Ca_junc + I_Ca_sl;

    d_finavalu[offset_43] =
        -I_Ca * Cmem / (Vmyo * (fp)2.0 * Frdy) * (fp)1e3;

    I_CaK = ibark * gate_common *
            (Fjunc_CaL * one_minus_fCa_j +
             Fsl_CaL   * one_minus_fCa_sl);

    I_CaNa_junc =
        Fjunc_CaL * ibarna_j * gate_common * one_minus_fCa_j;
    I_CaNa_sl =
        Fsl_CaL   * ibarna_sl * gate_common * one_minus_fCa_sl;

    // I_ncx: Na/Ca Exchanger flux
    const fp Kdact_over_Caj = Kdact / d_initvalu_36;
    const fp Kdact_over_Casl = Kdact / d_initvalu_37;
    Ka_junc = (fp)1.0 /
              ((fp)1.0 + pow(Kdact_over_Caj, (fp)3.0));
    Ka_sl   = (fp)1.0 /
              ((fp)1.0 + pow(Kdact_over_Casl, (fp)3.0));

    const fp Nai_j3 = d_initvalu_32 * d_initvalu_32 * d_initvalu_32;
    const fp Nai_sl3 = d_initvalu_33 * d_initvalu_33 * d_initvalu_33;
    const fp Nao3 = Nao * Nao * Nao;

    const fp exp_nu_v = exp(nu * v39FoRT);
    const fp exp_nu_1_v = exp((nu - (fp)1.0) * v39FoRT);

    s1_junc = exp_nu_v * Nai_j3 * Cao;
    s1_sl   = exp_nu_v * Nai_sl3 * Cao;
    s2_junc = exp_nu_1_v * Nao3 * d_initvalu_36;
    s2_sl   = exp_nu_1_v * Nao3 * d_initvalu_37;

    const fp KmNai3 = KmNai * KmNai * KmNai;
    const fp KmNao3 = KmNao * KmNao * KmNao;

    s3_junc =
        (KmCai * Nao3 * ((fp)1.0 + pow(d_initvalu_32 / KmNai, (fp)3.0)) +
         KmNao3 * d_initvalu_36 +
         KmNai3 * Cao * ((fp)1.0 + d_initvalu_36 / KmCai) +
         KmCao * Nai_j3 + Nai_j3 * Cao +
         Nao3 * d_initvalu_36) *
        ((fp)1.0 + ksat * exp_nu_1_v);

    s3_sl =
        (KmCai * Nao3 * ((fp)1.0 + pow(d_initvalu_33 / KmNai, (fp)3.0)) +
         KmNao3 * d_initvalu_37 +
         KmNai3 * Cao * ((fp)1.0 + d_initvalu_37 / KmCai) +
         KmCao * Nai_sl3 + Nai_sl3 * Cao +
         Nao3 * d_initvalu_37) *
        ((fp)1.0 + ksat * exp_nu_1_v);

    const fp powQ10NCX_Qpow = pow(Q10NCX, Qpow);

    I_ncx_junc = Fjunc * IbarNCX * powQ10NCX_Qpow * Ka_junc *
                 (s1_junc - s2_junc) / s3_junc;
    I_ncx_sl   = Fsl   * IbarNCX * powQ10NCX_Qpow * Ka_sl *
                 (s1_sl - s2_sl) / s3_sl;
    I_ncx      = I_ncx_junc + I_ncx_sl;
    d_finavalu[offset_45] =
        (fp)2.0 * I_ncx * Cmem / (Vmyo * (fp)2.0 * Frdy) * (fp)1e3;

    // I_pca: Sarcolemmal Ca Pump Current
    const fp powQ10SLCaP_Qpow = pow(Q10SLCaP, Qpow);
    const fp exp16 = (fp)1.6;
    const fp Ca_j16  = pow(d_initvalu_36, exp16);
    const fp Ca_sl16 = pow(d_initvalu_37, exp16);
    const fp KmPCa16 = pow(KmPCa, exp16);

    I_pca_junc = Fjunc * powQ10SLCaP_Qpow * IbarSLCaP *
                 Ca_j16 / (KmPCa16 + Ca_j16);
    I_pca_sl   = Fsl   * powQ10SLCaP_Qpow * IbarSLCaP *
                 Ca_sl16 / (KmPCa16 + Ca_sl16);
    I_pca = I_pca_junc + I_pca_sl;
    d_finavalu[offset_44] =
        -I_pca * Cmem / (Vmyo * (fp)2.0 * Frdy) * (fp)1e3;

    // I_cabk: Ca Background Current
    I_cabk_junc = Fjunc * GCaB * (v39 - eca_junc);
    I_cabk_sl   = Fsl   * GCaB * (v39 - eca_sl);
    I_cabk      = I_cabk_junc + I_cabk_sl;
    d_finavalu[offset_46] =
        -I_cabk * Cmem / (Vmyo * (fp)2.0 * Frdy) * (fp)1e3;

    // SR fluxes: Calcium Release, SR Ca pump, SR Ca leak
    MaxSR = (fp)15.0;
    MinSR = (fp)1.0;
    kCaSR =
        MaxSR - (MaxSR - MinSR) /
                    ((fp)1.0 + pow(ec50SR / d_initvalu_31, (fp)2.5));
    koSRCa = koCa / kCaSR;
    kiSRCa = kiCa * kCaSR;
    RI = (fp)1.0 - d_initvalu_14 - d_initvalu_15 - d_initvalu_16;

    const fp Ca_j2 = d_initvalu_36 * d_initvalu_36;

    d_finavalu[offset_14] =
        (kim * RI - kiSRCa * d_initvalu_36 * d_initvalu_14) -
        (koSRCa * Ca_j2 * d_initvalu_14 - kom * d_initvalu_15);
    d_finavalu[offset_15] =
        (koSRCa * Ca_j2 * d_initvalu_14 - kom * d_initvalu_15) -
        (kiSRCa * d_initvalu_36 * d_initvalu_15 - kim * d_initvalu_16);
    d_finavalu[offset_16] =
        (kiSRCa * d_initvalu_36 * d_initvalu_15 - kim * d_initvalu_16) -
        (kom * d_initvalu_16 - koSRCa * Ca_j2 * RI);

    J_SRCarel = ks * d_initvalu_15 *
                (d_initvalu_31 - d_initvalu_36);
    const fp powQ10SRCaP_Qpow = pow(Q10SRCaP, Qpow);
    const fp Ca_i_over_Kmf = d_initvalu_38 / Kmf;
    const fp Ca_sr_over_Kmr = d_initvalu_31 / Kmr;
    const fp Ca_i_over_Kmf_pow =
        pow(Ca_i_over_Kmf, hillSRCaP);
    const fp Ca_sr_over_Kmr_pow =
        pow(Ca_sr_over_Kmr, hillSRCaP);

    J_serca = powQ10SRCaP_Qpow * Vmax_SRCaP *
              (Ca_i_over_Kmf_pow - Ca_sr_over_Kmr_pow) /
              ((fp)1.0 + Ca_i_over_Kmf_pow + Ca_sr_over_Kmr_pow);

    J_SRleak = (fp)5.348e-6 *
               (d_initvalu_31 - d_initvalu_36);

    // Sodium and Calcium Buffering
    d_finavalu[offset_17] =
        kon_na * d_initvalu_32 *
            (Bmax_Naj - d_initvalu_17) -
        koff_na * d_initvalu_17;
    d_finavalu[offset_18] =
        kon_na * d_initvalu_33 *
            (Bmax_Nasl - d_initvalu_18) -
        koff_na * d_initvalu_18;

    // Cytosolic Ca Buffers
    d_finavalu[offset_19] =
        kon_tncl * d_initvalu_38 *
            (Bmax_TnClow - d_initvalu_19) -
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
        kon_sr * d_initvalu_38 *
            (Bmax_SR - d_initvalu_25) -
        koff_sr * d_initvalu_25;

    J_CaB_cytosol =
        d_finavalu[offset_19] + d_finavalu[offset_20] +
        d_finavalu[offset_21] + d_finavalu[offset_22] +
        d_finavalu[offset_23] + d_finavalu[offset_24] +
        d_finavalu[offset_25];

    // Junctional and SL Ca Buffers
    d_finavalu[offset_26] =
        kon_sll * d_initvalu_36 *
            (Bmax_SLlowj - d_initvalu_26) -
        koff_sll * d_initvalu_26;
    d_finavalu[offset_27] =
        kon_sll * d_initvalu_37 *
            (Bmax_SLlowsl - d_initvalu_27) -
        koff_sll * d_initvalu_27;
    d_finavalu[offset_28] =
        kon_slh * d_initvalu_36 *
            (Bmax_SLhighj - d_initvalu_28) -
        koff_slh * d_initvalu_28;
    d_finavalu[offset_29] =
        kon_slh * d_initvalu_37 *
            (Bmax_SLhighsl - d_initvalu_29) -
        koff_slh * d_initvalu_29;

    J_CaB_junction = d_finavalu[offset_26] + d_finavalu[offset_28];
    J_CaB_sl       = d_finavalu[offset_27] + d_finavalu[offset_29];

    // SR Ca Concentrations
    d_finavalu[offset_30] =
        kon_csqn * d_initvalu_31 *
            (Bmax_Csqn - d_initvalu_30) -
        koff_csqn * d_initvalu_30;

    oneovervsr = (fp)1.0 / Vsr;
    d_finavalu[offset_31] =
        J_serca * Vmyo * oneovervsr -
        (J_SRleak * Vmyo * oneovervsr + J_SRCarel) -
        d_finavalu[offset_30];

    // Sodium Concentrations
    I_Na_tot_junc =
        I_Na_junc + I_nabk_junc + (fp)3.0 * I_ncx_junc +
        (fp)3.0 * I_nak_junc + I_CaNa_junc;
    I_Na_tot_sl =
        I_Na_sl + I_nabk_sl + (fp)3.0 * I_ncx_sl +
        (fp)3.0 * I_nak_sl + I_CaNa_sl;

    d_finavalu[offset_32] =
        -I_Na_tot_junc * Cmem / (Vjunc * Frdy) +
        J_na_juncsl / Vjunc *
            (d_initvalu_33 - d_initvalu_32) -
        d_finavalu[offset_17];

    oneovervsl = (fp)1.0 / Vsl;
    d_finavalu[offset_33] =
        -I_Na_tot_sl * Cmem * oneovervsl / Frdy +
        J_na_juncsl * oneovervsl *
            (d_initvalu_32 - d_initvalu_33) +
        J_na_slmyo * oneovervsl *
            (d_initvalu_34 - d_initvalu_33) -
        d_finavalu[offset_18];

    d_finavalu[offset_34] =
        J_na_slmyo / Vmyo *
        (d_initvalu_33 - d_initvalu_34);

    // Potassium Concentration
    I_K_tot =
        I_to + I_kr + I_ks + I_ki -
        (fp)2.0 * I_nak + I_CaK + I_kp;
    d_finavalu[offset_35] = (fp)0.0;

    // Calcium Concentrations
    I_Ca_tot_junc =
        I_Ca_junc + I_cabk_junc + I_pca_junc -
        (fp)2.0 * I_ncx_junc;
    I_Ca_tot_sl =
        I_Ca_sl + I_cabk_sl + I_pca_sl -
        (fp)2.0 * I_ncx_sl;

    d_finavalu[offset_36] =
        -I_Ca_tot_junc * Cmem /
            (Vjunc * (fp)2.0 * Frdy) +
        J_ca_juncsl / Vjunc *
            (d_initvalu_37 - d_initvalu_36) -
        J_CaB_junction +
        J_SRCarel * Vsr / Vjunc +
        J_SRleak * Vmyo / Vjunc;

    d_finavalu[offset_37] =
        -I_Ca_tot_sl * Cmem /
            (Vsl * (fp)2.0 * Frdy) +
        J_ca_juncsl / Vsl *
            (d_initvalu_36 - d_initvalu_37) +
        J_ca_slmyo / Vsl *
            (d_initvalu_38 - d_initvalu_37) -
        J_CaB_sl;

    d_finavalu[offset_38] =
        -J_serca - J_CaB_cytosol +
        J_ca_slmyo / Vmyo *
            (d_initvalu_37 - d_initvalu_38);

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
        if (timeinst > (fp)0.5 & timeinst < (fp)200.5) {
            V_clamp = V_test;
        } else {
            V_clamp = V_hold;
        }
        R_clamp = (fp)0.04;
        I_app = (V_clamp - v39) / R_clamp;
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
