__global__ void srad2(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                      int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                      fp *d_dE, fp *d_dW, fp *d_c, fp *d_I) {

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

    // Compute indices for neighbors
    int idx_S = d_iS[row] + d_Nr * col;
    int idx_E = row + d_Nr * d_jE[col];

    // Load diffusion coefficients with coalesced access
    fp d_cN = d_c[ei];
    fp d_cS = d_c[idx_S];
    fp d_cW = d_c[ei];
    fp d_cE = d_c[idx_E];

    // Load directional derivatives
    fp d_dN_loc = d_dN[ei];
    fp d_dS_loc = d_dS[ei];
    fp d_dW_loc = d_dW[ei];
    fp d_dE_loc = d_dE[ei];

    // Compute divergence using fused multiply-add
    fp d_D = fmaf(d_cN, d_dN_loc, fmaf(d_cS, d_dS_loc, 
                  fmaf(d_cW, d_dW_loc, d_cE * d_dE_loc)));

    // Compute update factor
    fp update_factor = 0.25f * d_lambda * d_D;

    // Load current image value and update
    fp d_I_current = d_I[ei];
    d_I[ei] = d_I_current + update_factor;
}
