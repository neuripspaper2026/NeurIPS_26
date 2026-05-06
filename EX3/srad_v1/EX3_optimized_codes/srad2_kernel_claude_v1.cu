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

    // Compute indices for neighbors (matching srad_kernel pattern)
    int idx_S = d_iS[row] + d_Nr * col;
    int idx_E = row + d_Nr * d_jE[col];

    // Load diffusion coefficients with coalesced access
    fp d_cN = d_c[ei];
    fp d_cS = d_c[idx_S];
    fp d_cW = d_c[ei];
    fp d_cE = d_c[idx_E];

    // Load directional derivatives
    fp d_dN_val = d_dN[ei];
    fp d_dS_val = d_dS[ei];
    fp d_dW_val = d_dW[ei];
    fp d_dE_val = d_dE[ei];

    // Compute divergence using FMA operations
    fp d_D = d_cN * d_dN_val + d_cS * d_dS_val + d_cW * d_dW_val + d_cE * d_dE_val;

    // Precompute constant factor
    fp update_factor = fp(0.25) * d_lambda;

    // Load current image value
    fp d_I_val = d_I[ei];

    // Image update with FMA
    d_I_val = d_I_val + update_factor * d_D;

    // Coalesced write to global memory
    d_I[ei] = d_I_val;
}
