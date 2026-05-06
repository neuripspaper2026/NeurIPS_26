__global__ void srad(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                     int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                     fp *d_dE, fp *d_dW, fp d_q0sqr, fp *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ei = bx * NUMBER_THREADS + tx;

    if (ei >= d_Ne) return;

    // Calculate row/col using optimized integer arithmetic
    int row = ei % d_Nr;
    int col = ei / d_Nr;

    // Load current element value once
    fp d_Jc = d_I[ei];
    
    // Precompute common index calculations
    int row_iN = d_iN[row];
    int row_iS = d_iS[row];
    int col_jW = d_jW[col];
    int col_jE = d_jE[col];
    
    int base_col = d_Nr * col;
    
    // Coalesced memory access pattern - load neighboring values
    fp I_N = d_I[row_iN + base_col];
    fp I_S = d_I[row_iS + base_col];
    fp I_W = d_I[row + d_Nr * col_jW];
    fp I_E = d_I[row + d_Nr * col_jE];

    // Compute directional derivatives
    fp d_dN_loc = I_N - d_Jc;
    fp d_dS_loc = I_S - d_Jc;
    fp d_dW_loc = I_W - d_Jc;
    fp d_dE_loc = I_E - d_Jc;

    // Compute squared derivatives for gradient calculation
    fp dN_sq = d_dN_loc * d_dN_loc;
    fp dS_sq = d_dS_loc * d_dS_loc;
    fp dW_sq = d_dW_loc * d_dW_loc;
    fp dE_sq = d_dE_loc * d_dE_loc;

    // Normalized discrete gradient magnitude squared (equ 52,53)
    fp d_Jc_sq = d_Jc * d_Jc;
    fp d_G2 = (dN_sq + dS_sq + dW_sq + dE_sq) / d_Jc_sq;

    // Normalized discrete laplacian (equ 54)
    fp d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;

    // ICOV (equ 31/35) - optimized computation
    fp d_L_sq = d_L * d_L;
    fp d_num = fmaf(-0.0625f, d_L_sq, 0.5f * d_G2);
    fp d_den = fmaf(0.25f, d_L, 1.0f);
    fp d_den_sq = d_den * d_den;
    fp d_qsqr = d_num / d_den_sq;

    // Diffusion coefficient (equ 33)
    fp d_q0sqr_term = d_q0sqr * fmaf(d_q0sqr, 1.0f, 1.0f);
    fp d_den_coeff = (d_qsqr - d_q0sqr) / d_q0sqr_term;
    fp d_c_loc = 1.0f / fmaf(d_den_coeff, 1.0f, 1.0f);

    // Saturate diffusion coefficient to [0, 1] range using fmin/fmax
    d_c_loc = fminf(fmaxf(d_c_loc, 0.0f), 1.0f);

    // Coalesced writes to global memory
    d_dN[ei] = d_dN_loc;
    d_dS[ei] = d_dS_loc;
    d_dW[ei] = d_dW_loc;
    d_dE[ei] = d_dE_loc;
    d_c[ei] = d_c_loc;
}
