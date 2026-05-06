__global__ void srad(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                     int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                     fp *d_dE, fp *d_dW, fp d_q0sqr, fp *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ei = bx * NUMBER_THREADS + tx;

    if (ei >= d_Ne) return;

    // figure out row/col location in new matrix
    int row, col;
    int ei_plus_1 = ei + 1;
    int remainder = ei_plus_1 % d_Nr;
    
    if (remainder == 0) {
        row = d_Nr - 1;
        col = ei_plus_1 / d_Nr - 1;
    } else {
        row = remainder - 1;
        col = ei_plus_1 / d_Nr;
    }

    // Precompute base indices for better memory access
    int row_base = row;
    int col_base_Nr = d_Nr * col;

    // Load current element
    fp d_Jc = d_I[ei];
    fp d_Jc_inv = 1.0f / d_Jc;
    fp d_Jc_sq_inv = d_Jc_inv * d_Jc_inv;

    // Load neighbor indices
    int iN = d_iN[row];
    int iS = d_iS[row];
    int jW = d_jW[col];
    int jE = d_jE[col];

    // Compute directional derivatives with coalesced memory access
    fp d_dN_loc = d_I[iN + col_base_Nr] - d_Jc;
    fp d_dS_loc = d_I[iS + col_base_Nr] - d_Jc;
    fp d_dW_loc = d_I[row_base + d_Nr * jW] - d_Jc;
    fp d_dE_loc = d_I[row_base + d_Nr * jE] - d_Jc;

    // Compute gradient squared (normalized)
    fp d_G2 = (d_dN_loc * d_dN_loc + d_dS_loc * d_dS_loc +
               d_dW_loc * d_dW_loc + d_dE_loc * d_dE_loc) * d_Jc_sq_inv;

    // Compute laplacian (normalized)
    fp d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) * d_Jc_inv;

    // ICOV computation
    fp d_L_sq = d_L * d_L;
    fp d_num = 0.5f * d_G2 - 0.0625f * d_L_sq;
    fp d_den = 1.0f + 0.25f * d_L;
    fp d_qsqr = d_num / (d_den * d_den);

    // Diffusion coefficient
    fp d_den2 = (d_qsqr - d_q0sqr) / (d_q0sqr * (1.0f + d_q0sqr));
    fp d_c_loc = 1.0f / (1.0f + d_den2);

    // Clamp diffusion coefficient to [0, 1]
    d_c_loc = fminf(fmaxf(d_c_loc, 0.0f), 1.0f);

    // Coalesced writes to global memory
    d_dN[ei] = d_dN_loc;
    d_dS[ei] = d_dS_loc;
    d_dW[ei] = d_dW_loc;
    d_dE[ei] = d_dE_loc;
    d_c[ei] = d_c_loc;
}
