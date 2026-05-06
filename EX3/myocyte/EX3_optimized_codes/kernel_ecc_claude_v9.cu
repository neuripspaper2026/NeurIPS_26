<<<CODE>>>
__device__ void kernel_ecc(fp timeinst, fp *d_initvalu, fp *d_finavalu,
                           int valu_offset, fp *d_params) {

    //=====================================================================
    //	VARIABLES
    //=====================================================================

    // input parameters
    fp cycleLength;

    // matlab constants undefined in c
    const fp pi = 3.1416;

    // Constants
    const fp R = 8314;               // [J/kmol*K]
    const fp Frdy = 96485;           // [C/mol]
    const fp Temp = 310;             // [K] 310
    const fp FoRT = Frdy / R / Temp; //
    const fp Cmem = 1.3810e-10;      // [F] membrane capacitance
    const fp Qpow = (Temp - 310) / 10;

    // Cell geometry
    const fp cellLength = 100;   // cell length [um]
    const fp cellRadius = 10.25; // cell radius [um]
    const fp Vcell = pi * cellRadius * cellRadius * cellLength * 1e-15; // [L]
    const fp Vmyo = 0.65 * Vcell;
    const fp Vsr = 0.035 * Vcell;
    const fp Vsl = 0.02 * Vcell;
    const fp Vjunc = 0.0539 * 0.01 * Vcell;
    const fp J_ca_juncsl = 1 / 1.2134e12;             // [L/msec]
    const fp J_ca_slmyo = 1 / 2.68510e11;             // [L/msec]
    const fp J_na_juncsl = 1 / (1.6382e12 / 3 * 100); // [L/msec]
    const fp J_na_slmyo = 1 / (1.8308e10 / 3 * 100);  // [L/msec]

    // Fractional currents in compartments
    const fp Fjunc = 0.11;
    const fp Fsl = 1 - Fjunc;
    const fp Fjunc_CaL = 0.9;
    const fp Fsl_CaL = 1 - Fjunc_CaL;

    // Fixed ion concentrations
    const fp Cli = 15;  // Intracellular Cl  [mM]
    const fp Clo = 150; // Extracellular Cl  [mM]
    const fp Ko = 5.4;  // Extracellular K   [mM]
    const fp Nao = 140; // Extracellular Na  [mM]
    const fp Cao = 1.8; // Extracellular Ca  [mM]
    const fp Mgi = 1;   // Intracellular Mg  [mM]

    // Na transport parameters
    const fp GNa = 16.0;        // [mS/uF]
    const fp GNaB = 0.297e-3;   // [mS/uF]
    const fp IbarNaK = 1.90719; // [uA/uF]
    const fp KmNaip = 11;       // [mM]
    const fp KmKo = 1.5;        // [mM]

    // K current parameters
    const fp pNaK = 0.01833;
    const fp GtoSlow = 0.06; // [mS/uF]
    const fp GtoFast = 0.02; // [mS/uF]
    const fp gkp = 0.001;

    // Cl current parameters
    const fp GClCa = 0.109625; // [mS/uF]
    const fp GClB = 9e-3;      // [mS/uF]
    const fp KdClCa = 100e-3;  // [mM]

    // I_Ca parameters
    const fp pNa = 1.5e-8; // [cm/sec]
    const fp pCa = 5.4e-4; // [cm/sec]
    const fp pK = 2.7e-7;  // [cm/sec]
    const fp Q10CaL = 1.8;

    // Ca transport parameters
    const fp IbarNCX = 9.0;      // [uA/uF]
    const fp KmCai = 3.59e-3;    // [mM]
    const fp KmCao = 1.3;        // [mM]
    const fp KmNai = 12.29;      // [mM]
    const fp KmNao = 87.5;       // [mM]
    const fp ksat = 0.27;        // [none]
    const fp nu = 0.35;          // [none]
    const fp Kdact = 0.256e-3;   // [mM]
    const fp Q10NCX = 1.57;      // [none]
    const fp IbarSLCaP = 0.0673; // [uA/uF]
    const fp KmPCa = 0.5e-3;     // [mM]
    const fp GCaB = 2.513e-4;    // [uA/uF]
    const fp Q10SLCaP = 2.35;    // [none]

    // SR flux parameters
    const fp Q10SRCaP = 2.6;       // [none]
    const fp Vmax_SRCaP = 2.86e-4; // [mM/msec] (mmol/L cytosol/msec)
    const fp Kmf = 0.246e-3;       // [mM]
    const fp Kmr = 1.7;            // [mM]L cytosol
    const fp hillSRCaP = 1.787;    // [mM]
    const fp ks = 25;              // [1/ms]
    const fp koCa = 10;            // [mM^-2 1/ms]
    const fp kom = 0.06;           // [1/ms]
    const fp kiCa = 0.5;           // [1/mM/ms]
    const fp kim = 0.005;          // [1/ms]
    const fp ec50SR = 0.45;        // [mM]

    // Buffering parameters
    const fp Bmax_Naj = 7.561;       // [mM]
    const fp Bmax_Nasl = 1.65;       // [mM]
    const fp koff_na = 1e-3;         // [1/ms]
    const fp kon_na = 0.1e-3;        // [1/mM/ms]
    const fp Bmax_TnClow = 70e-3;    // [mM], TnC low affinity
    const fp koff_tncl = 19.6e-3;    // [1/ms]
    const fp kon_tncl = 32.7;        // [1/mM/ms]
    const fp Bmax_TnChigh = 140e-3;  // [mM], TnC high affinity
    const fp koff_tnchca = 0.032e-3; // [1/ms]
    const fp kon_tnchca = 2.37;      // [1/mM/ms]
    const fp koff_tnchmg = 3.33e-3;  // [1/ms]
    const fp kon_tnchmg = 3e-3;      // [1/mM/ms]
    const fp Bmax_myosin = 140e-3;                        // [mM], Myosin buffering
    const fp koff_myoca = 0.46e-3;                        // [1/ms]
    const fp kon_myoca = 13.8;                            // [1/mM/ms]
    const fp koff_myomg = 0.057e-3;                       // [1/ms]
    const fp kon_myomg = 0.0157;                          // [1/mM/ms]
    const fp Bmax_SR = 19 * 0.9e-3;                       // [mM]
    const fp koff_sr = 60e-3;                             // [1/ms]
    const fp kon_sr = 100;                                // [1/mM/ms]
    const fp Bmax_SLlowsl = 37.38e-3 * Vmyo / Vsl;        // [mM], SL buffering
    const fp Bmax_SLlowj = 4.62e-3 * Vmyo / Vjunc * 0.1;  // [mM]
    const fp koff_sll = 1300e-3;                          // [1/ms]
    const fp kon_sll = 100;                               // [1/mM/ms]
    const fp Bmax_SLhighsl = 13.35e-3 * Vmyo / Vsl;       // [mM]
    const fp Bmax_SLhighj = 1.65e-3 * Vmyo / Vjunc * 0.1; // [mM]
    const fp koff_slh = 30e-3;                            // [1/ms]
    const fp kon_slh = 100;                               // [1/mM/ms]
    const fp Bmax_Csqn = 2.7;                             // 140e-3*Vmyo/Vsr; [mM]
    const fp koff_csqn = 65;                              // [1/ms]
    const fp kon_csqn = 100;                              // [1/mM/ms]

    // Load input parameters
    cycleLength = d_params[15];

    // Load state variables from global memory (coalesced reads)
    fp d_initvalu_1 = d_initvalu[valu_offset];
    fp d_initvalu_2 = d_initvalu[valu_offset + 1];
    fp d_initvalu_3 = d_initvalu[valu_offset + 2];
    fp d_initvalu_4 = d_initvalu[valu_offset + 3];
    fp d_initvalu_5 = d_initvalu[valu_offset + 4];
    fp d_initvalu_6 = d_initvalu[valu_offset + 5];
    fp d_initvalu_7 = d_initvalu[valu_offset + 6];
    fp d_initvalu_8 = d_initvalu[valu_offset + 7];
    fp d_initvalu_9 = d_initvalu[valu_offset + 8];
    fp d_initvalu_10 = d_initvalu[valu_offset + 9];
    fp d_initvalu_11 = d_initvalu[valu_offset + 10];
    fp d_initvalu_12 = d_initvalu[valu_offset + 11];
    fp d_initvalu_13 = d_initvalu[valu_offset + 12];
    fp d_initvalu_14 = d_initvalu[valu_offset + 13];
    fp d_initvalu_15 = d_initvalu[valu_offset + 14];
    fp d_initvalu_16 = d_initvalu[valu_offset + 15];
    fp d_initvalu_17 = d_initvalu[valu_offset + 16];
    fp d_initvalu_18 = d_initvalu[valu_offset + 17];
    fp d_initvalu_19 = d_initvalu[valu_offset + 18];
    fp d_initvalu_20 = d_initvalu[valu_offset + 19];
    fp d_initvalu_21 = d_initvalu[valu_offset + 20];
    fp d_initvalu_23 = d_initvalu[valu_offset + 22];
    fp d_initvalu_24 = d_initvalu[valu_offset + 23];
    fp d_initvalu_25 = d_initvalu[valu_offset + 24];
    fp d_initvalu_26 = d_initvalu[valu_offset + 25];
    fp d_initvalu_27 = d_initvalu[valu_offset + 26];
    fp d_initvalu_28 = d_initvalu[valu_offset + 27];
    fp d_initvalu_29 = d_initvalu[valu_offset + 28];
    fp d_initvalu_30 = d_initvalu[valu_offset + 29];
    fp d_initvalu_31 = d_initvalu[valu_offset + 30];
    fp d_initvalu_32 = d_initvalu[valu_offset + 31];
    fp d_initvalu_33 = d_initvalu[valu_offset + 32];
    fp d_initvalu_34 = d_initvalu[valu_offset + 33];
    fp d_initvalu_35 = d_initvalu[valu_offset + 34];
    fp d_initvalu_36 = d_initvalu[valu_offset + 35];
    fp d_initvalu_37 = d_initvalu[valu_offset + 36];
    fp d_initvalu_38 = d_initvalu[valu_offset + 37];
    fp d_initvalu_39 = d_initvalu[valu_offset + 38];
    fp d_initvalu_40 = d_initvalu[valu_offset + 39];

    // Nernst Potentials
    fp ena_junc = (1 / FoRT) * log(Nao / d_initvalu_32);
    fp ena_sl = (1 / FoRT) * log(Nao / d_initvalu_33);
    fp ek = (1 / FoRT) * log(Ko / d_initvalu_35);
    fp eca_junc = (1 / FoRT / 2) * log(Cao / d_initvalu_36);
    fp eca_sl = (1 / FoRT / 2) * log(Cao / d_initvalu_37);
    fp ecl = (1 / FoRT) * log(Cli / Clo);

    // I_Na: Fast Na Current
    fp am = 0.32 * (d_initvalu_39 + 47.13) / (1 - exp(-0.1 * (d_initvalu_39 + 47.13)));
    fp bm = 0.08 * exp(-d_initvalu_39 / 11);
    fp ah, bh, aj, bj;
    if (d_initvalu_39 >= -40) {
        ah = 0;
        aj = 0;
        bh = 1 / (0.13 * (1 + exp(-(d_initvalu_39 + 10.66) / 11.1)));
        bj = 0.3 * exp(-2.535e-7 * d_initvalu_39) / (1 + exp(-0.1 * (d_initvalu_39 + 32)));
    } else {
        ah = 0.135 * exp((80 + d_initvalu_39) / -6.8);
        bh = 3.56 * exp(0.079 * d_initvalu_39) + 3.1e5 * exp(0.35 * d_initvalu_39);
        aj = (-127140 * exp(0.2444 * d_initvalu_39) - 3.474e-5 * exp
