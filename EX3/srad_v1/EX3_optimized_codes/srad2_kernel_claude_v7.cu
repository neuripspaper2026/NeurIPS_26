__global__ void srad2(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                      int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                      fp *d_dE, fp *d_dW, fp *d_c, fp *d_I) {

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
    int col_base_Nr = d_Nr * col;
    int iS_index = d_iS[row] + col_base_Nr;
    int jE_index = row + d_Nr * d_jE[col];

    // Load diffusion coefficients with coalesced access pattern
    fp d_cN = d_c[ei];
    fp d_cS = d_c[iS_index];
    fp d_cW = d_c[ei];
    fp d_cE = d_c[jE_index];

    // Load directional derivatives with coalesced access
    fp d_dN_val = d_dN[ei];
    fp d_dS_val = d_dS[ei];
    fp d_dW_val = d_dW[ei];
    fp d_dE_val = d_dE[ei];

    // Compute divergence using FMA operations
    fp d_D = d_cN * d_dN_val + d_cS * d_dS_val + 
             d_cW * d_dW_val + d_cE * d_dE_val;

    // Image update with FMA optimization
    fp d_I_current = d_I[ei];
    fp d_I_updated = fmaf(0.25f * d_lambda, d_D, d_I_current);

    // Coalesced write to global memory
    d_I[ei] = d_I_updated;
}
