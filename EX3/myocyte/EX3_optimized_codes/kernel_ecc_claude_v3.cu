<<<CODE>>>
__device__ void kernel_ecc(fp timeinst, fp *d_initvalu, fp *d_finavalu,
                           int valu_offset, fp *d_params) {

    // Load input parameter
    const fp cycleLength = d_params[15];

    // Compute offsets once
    const int offset_1 = valu_offset;
    const int offset_2 = valu_offset + 1;
    const int offset_3 = valu_offset + 2;
    const int offset_4 = valu_offset + 3;
    const int offset_5 = valu_offset + 4;
    const int offset_6 = valu_offset + 5;
    const int offset_7 = valu_offset + 6;
    const int offset_8 = valu_offset + 7;
    const int offset_9 = valu_offset + 8;
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
    const int offset_43 = valu_offset + 42;
    const int offset_44 = valu_offset + 43;
    const int offset_45 = valu_offset + 44;
    const int offset_46 = valu_offset + 45;

    // Load state variables into registers (coalesced access)
    const fp d_initvalu_1 = d_initvalu[offset_1];
    const fp d_initvalu_2 = d_initvalu[offset_2];
    const fp d_initvalu_3 = d_initvalu[offset_3];
    const fp d_initvalu_4 = d_initvalu[offset_4];
    const fp d_initvalu_5 = d_initvalu[offset_5];
    const fp d_initvalu_6 = d_initvalu[offset_6];
    const fp d_initvalu_7 = d_initvalu[offset_7];
    const fp d_initvalu_8 = d_initvalu[offset_8];
    const fp d_initvalu_9 = d_initvalu[offset_9];
    const fp d_initvalu_10 = d_initvalu[offset_10];
    const fp d_initvalu_11 = d_initvalu[offset_11];
    const fp d_initvalu_12 = d_initvalu[offset_12];
    const fp d_initvalu_13 = d_initvalu[offset_13];
    const fp d_initvalu_14 = d_initvalu[offset_14];
    const fp d_initvalu_15 = d_initvalu[offset_15];
    const fp d_initvalu_16 = d_initvalu[offset_16];
    const fp d_initvalu_17 = d_initvalu[offset_17];
    const fp d_initvalu_18 = d_initvalu[offset_18];
    const fp d_initvalu_19 = d_initvalu[offset_19];
    const fp d_initvalu_20 = d_initvalu[offset_20];
    const fp d_initvalu_21 = d_initvalu[offset_21];
    const fp d_initvalu_23 = d_initvalu[offset_23];
    const fp d_initvalu_24 = d_initvalu[offset_24];
    const fp d_initvalu_25 = d_initvalu[offset_25];
    const fp d_initvalu_26 = d_initvalu[offset_26];
    const fp d_initvalu_27 = d_initvalu[offset_27];
    const fp d_initvalu_28 = d_initvalu[offset_28];
    const fp d_initvalu_29 = d_initvalu[offset_29];
    const fp d_initvalu_30 = d_initvalu[offset_30];
    const fp d_initvalu_31 = d_initvalu[offset_31];
    const fp d_initvalu_32 = d_initvalu[offset_32];
    const fp d_initvalu_33 = d_initvalu[offset_33];
    const fp d_initvalu_34 = d_initvalu[offset_34];
    const fp d_initvalu_35 = d_initvalu[offset_35];
    const fp d_initvalu_36 = d_initvalu[offset_36];
    const fp d_initvalu_37 = d_initvalu[offset_37];
    const fp d_initvalu_38 = d_initvalu[offset_38];
    const fp d_initvalu_39 = d_initvalu[offset_39];
    const fp d_initvalu_40 = d_initvalu[offset_40];

    // Constants
    const fp pi = 3.1416;
    const fp R = 8314;
    const fp Frdy = 96485;
    const fp Temp = 310;
    const fp FoRT = Frdy / R / Temp;
    const fp Cmem = 1.3810e-10;
    const fp Qpow = (Temp - 310) / 10;

    // Cell geometry
    const fp cellLength = 100;
    const fp cellRadius = 10.25;
    const fp Vcell = pi * cellRadius * cellRadius * cellLength * 1e-15;
    const fp Vmyo = 0.65 * Vcell;
    const fp Vsr = 0.035 * Vcell;
    const fp Vsl = 0.02 * Vcell;
    const fp Vjunc = 0.0539 * 0.01 * Vcell;
    const fp J_ca_juncsl = 1 / 1.2134e12;
    const fp J_ca_slmyo = 1 / 2.68510e11;
    const fp J_na_juncsl = 1 / (1.6382e12 / 3 * 100);
    const fp J_na_slmyo = 1 / (1.8308e10 / 3 * 100);

    // Fractional currents
    const fp Fjunc = 0.11;
    const fp Fsl = 1 - Fjunc;
    const fp Fjunc_CaL = 0.9;
    const fp Fsl_CaL = 1 - Fjunc_CaL;

    // Fixed ion concentrations
    const fp Cli = 15;
    const fp Clo = 150;
    const fp Ko = 5.4;
    const fp Nao = 140;
    const fp Cao = 1.8;
    const fp Mgi = 1;

    // Nernst Potentials
    const fp ena_junc = (1 / FoRT) * log(Nao / d_initvalu_32);
    const fp ena_sl = (1 / FoRT) * log(Nao / d_initvalu_33);
    const fp ek = (1 / FoRT) * log(Ko / d_initvalu_35);
    const fp eca_junc = (1 / FoRT / 2) * log(Cao / d_initvalu_36);
    const fp eca_sl = (1 / FoRT / 2) * log(Cao / d_initvalu_37);
    const fp ecl = (1 / FoRT) * log(Cli / Clo);

    // Na transport parameters
    const fp GNa = 16.0;
    const fp GNaB = 0.297e-3;
    const fp IbarNaK = 1.90719;
    const fp KmNaip = 11;
    const fp KmKo = 1.5;

    // K current parameters
    const fp pNaK = 0.01833;
    const fp GtoSlow = 0.06;
    const fp GtoFast = 0.02;
    const fp gkp = 0.001;

    // Cl current parameters
    const fp GClCa = 0.109625;
    const fp GClB = 9e-3;
    const fp KdClCa = 100e-3;

    // I_Ca parameters
    const fp pNa = 1.5e-8;
    const fp pCa = 5.4e-4;
    const fp pK = 2.7e-7;
    const fp Q10CaL = 1.8;

    // Ca transport parameters
    const fp IbarNCX = 9.0;
    const fp KmCai = 3.59e-3;
    const fp KmCao = 1.3;
    const fp KmNai = 12.29;
    const fp KmNao = 87.5;
    const fp ksat = 0.27;
    const fp nu = 0.35;
    const fp Kdact = 0.256e-3;
    const fp Q10NCX = 1.57;
    const fp IbarSLCaP = 0.0673;
    const fp KmPCa = 0.5e-3;
    const fp GCaB = 2.513e-4;
    const fp Q10SLCaP = 2.35;

    // SR flux parameters
    const fp Q10SRCaP = 2.6;
    const fp Vmax_SRCaP = 2.86e-4;
    const fp Kmf = 0.246e-3;
    const fp Kmr = 1.7;
    const fp hillSRCaP = 1.787;
    const fp ks = 25;
    const fp koCa = 10;
    const fp kom = 0.06;
    const fp kiCa = 0.5;
    const fp kim = 0.005;
    const fp ec50SR = 0.45;

    // Buffering parameters
    const fp Bmax_Naj = 7.561;
    const fp Bmax_Nasl = 1.65;
    const fp koff_na = 1e-3;
    const fp kon_na = 0.1e-3;
    const fp Bmax_TnClow = 70e-3;
    const fp koff_tncl = 19.6e-3;
    const fp kon_tncl = 32.7;
    const fp Bmax_TnChigh = 140e-3;
    const fp koff_tnchca = 0.032e-3;
    const fp kon_tnchca = 2.37;
    const fp koff_tnchmg = 3.33e-3;
    const fp kon_tnchmg = 3e-3;
    const fp Bmax_myosin = 140e-3;
    const fp koff_myoca = 0.46e-3;
    const fp kon_myoca = 13.8;
    const fp koff_myomg = 0.057e-3;
    const fp kon_myomg = 0.0157;
    const fp Bmax_SR = 19 * 0.9e-3;
    const fp koff_sr = 60e-3;
    const fp kon_sr = 100;
    const fp Bmax_SLlowsl = 37.38e-3 * Vmyo / Vsl;
    const fp Bmax_SLlowj = 4.62e-3 * Vmyo / Vjunc * 0.1;
    const fp koff_sll = 1300e-3;
    const fp kon_sll = 100;
    const fp Bmax_SLhighsl = 13.35e-3 * Vmyo / Vsl;
    const fp Bmax_SLhighj = 1.65e-3 * Vmyo / Vjunc * 0.1;
    const fp koff_slh = 30e-3;
    const fp kon_slh = 100;
    const fp Bmax_Csqn = 2.7;
    const fp koff_csqn = 65;
    const fp kon_csqn = 100;

    // Precompute powers
    const fp Q10CaL_Qpow = pow(Q10CaL, Qpow);
    const fp Q10NCX_Qpow = pow(Q10NCX, Qpow);
    const fp Q10SLCaP_Qpow = pow(Q10SLCaP, Qpow);
    const fp Q10SRCaP_Qpow = pow(Q10SRCaP, Qpow);

    // I_Na: Fast Na Current
    const fp am = 0.32 * (d_initvalu_39 + 47.13) / (1 - exp(-0.1 * (d_initvalu_39 + 47.13)));
    const fp bm = 0.08 * exp(-d_initvalu_39 / 11);
    
    fp ah, bh, aj, bj;
    if (d_initvalu_39 >= -40) {
        ah = 0;
        aj = 0;
        bh = 1 / (0.13 * (1 + exp(-(d_initvalu_39 + 10.66) / 11.1)));
        bj = 0.3 * exp(-2.535e-7 * d_initvalu_39) / (1 + exp(-0.1 * (d_initvalu_39 + 32)));
    } else {
        ah = 0.135 * exp((80 + d_initvalu_39) / -6.8);
        bh = 3.56 * exp(0.079 * d_initvalu_39) + 3.1e5 * exp(0.35 * d_initvalu_39);
        aj = (-127140 * exp(0.2444 * d_
