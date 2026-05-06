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
    fp d_Jc_inv = fp(1.0) / d_Jc;

    // Compute indices once
    int idx_N = d_iN[row] + d_Nr * col;
    int idx_S = d_iS[row] + d_Nr * col;
    int idx_W = row + d_Nr * d_jW[col];
    int idx_E = row + d_Nr * d_jE[col];

    // directional derivatives with coalesced loads
    fp d_dN_loc = d_I[idx_N] - d_Jc;
    fp d_dS_loc = d_I[idx_S] - d_Jc;
    fp d_dW_loc = d_I[idx_W] - d_Jc;
    fp d_dE_loc = d_I[idx_E] - d_Jc;

    // normalized discrete gradient mag squared (equ 52,53)
    fp d_G2 = (d_dN_loc * d_dN_loc + d_dS_loc * d_dS_loc +
               d_dW_loc * d_dW_loc + d_dE_loc * d_dE_loc) *
              (d_Jc_inv * d_Jc_inv);

    // normalized discrete laplacian (equ 54)
    fp d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) * d_Jc_inv;

    // ICOV (equ 31/35)
    fp d_L_sq = d_L * d_L;
    fp d_num = fmaf(fp(-0.0625), d_L_sq, fp(0.5) * d_G2);
    fp d_den = fmaf(fp(0.25), d_L, fp(1.0));
    fp d_den_inv = fp(1.0) / d_den;
    fp d_qsqr = d_num * d_den_inv * d_den_inv;

    // diffusion coefficent (equ 33)
    fp d_q0sqr_term = d_q0sqr * (fp(1.0) + d_q0sqr);
    fp d_den2 = (d_qsqr - d_q0sqr) / d_q0sqr_term;
    fp d_c_loc = fp(1.0) / (fp(1.0) + d_den2);

    // saturate diffusion coefficent to 0-1 range
    d_c_loc = fmaxf(fp(0.0), fminf(fp(1.0), d_c_loc));

    // Coalesced writes to global memory
    d_dN[ei] = d_dN_loc;
    d_dS[ei] = d_dS_loc;
    d_dW[ei] = d_dW_loc;
    d_dE[ei] = d_dE_loc;
    d_c[ei] = d_c_loc;
}
