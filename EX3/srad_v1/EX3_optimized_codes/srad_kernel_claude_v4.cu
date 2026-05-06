__global__ void srad(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                     int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                     fp *d_dE, fp *d_dW, fp d_q0sqr, fp *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ei = bx * NUMBER_THREADS + tx;

    if (ei >= d_Ne) return;

    // figure out row/col location in new matrix
    int row = (ei + 1) % d_Nr - 1;
    int col = (ei + 1) / d_Nr + 1 - 1;
    if ((ei + 1) % d_Nr == 0) {
        row = d_Nr - 1;
        col = col - 1;
    }

    // Load current element value
    fp d_Jc = d_I[ei];
    fp d_Jc_inv = 1.0f / d_Jc;

    // Compute indices for neighbors
    int idx_N = d_iN[row] + d_Nr * col;
    int idx_S = d_iS[row] + d_Nr * col;
    int idx_W = row + d_Nr * d_jW[col];
    int idx_E = row + d_Nr * d_jE[col];

    // Load neighbor values
    fp I_N = d_I[idx_N];
    fp I_S = d_I[idx_S];
    fp I_W = d_I[idx_W];
    fp I_E = d_I[idx_E];

    // Compute directional derivatives
    fp d_dN_loc = I_N - d_Jc;
    fp d_dS_loc = I_S - d_Jc;
    fp d_dW_loc = I_W - d_Jc;
    fp d_dE_loc = I_E - d_Jc;

    // Compute gradient squared using fused multiply-add
    fp d_G2 = (d_dN_loc * d_dN_loc + d_dS_loc * d_dS_loc +
               d_dW_loc * d_dW_loc + d_dE_loc * d_dE_loc) *
              (d_Jc_inv * d_Jc_inv);

    // Compute laplacian
    fp d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) * d_Jc_inv;

    // ICOV computation
    fp d_L_sqr = d_L * d_L;
    fp d_num = fmaf(0.5f, d_G2, -0.0625f * d_L_sqr);
    fp d_den = fmaf(0.25f, d_L, 1.0f);
    fp d_den_sqr = d_den * d_den;
    fp d_qsqr = d_num / d_den_sqr;

    // Diffusion coefficient computation
    fp d_q0sqr_factor = d_q0sqr * (1.0f + d_q0sqr);
    fp d_den_c = (d_qsqr - d_q0sqr) / d_q0sqr_factor;
    fp d_c_loc = 1.0f / (1.0f + d_den_c);

    // Saturate diffusion coefficient to [0, 1]
    d_c_loc = fminf(fmaxf(d_c_loc, 0.0f), 1.0f);

    // Store results with coalesced writes
    d_dN[ei] = d_dN_loc;
    d_dS[ei] = d_dS_loc;
    d_dW[ei] = d_dW_loc;
    d_dE[ei] = d_dE_loc;
    d_c[ei] = d_c_loc;
}
