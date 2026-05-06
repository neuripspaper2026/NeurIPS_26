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

    // Compute indices once
    int idx_S = d_iS[row] + d_Nr * col;
    int idx_E = row + d_Nr * d_jE[col];

    // Coalesced loads of diffusion coefficients
    fp d_cN = d_c[ei];
    fp d_cS = d_c[idx_S];
    fp d_cW = d_c[ei];
    fp d_cE = d_c[idx_E];

    // Coalesced loads of directional derivatives
    fp dN = d_dN[ei];
    fp dS = d_dS[ei];
    fp dW = d_dW[ei];
    fp dE = d_dE[ei];

    // divergence computation using FMA operations
    fp d_D = fmaf(d_cN, dN, fmaf(d_cS, dS, fmaf(d_cW, dW, d_cE * dE)));

    // Load current image value
    fp I_curr = d_I[ei];

    // image update using FMA
    fp I_new = fmaf(0.25 * d_lambda, d_D, I_curr);

    // Coalesced store to global memory
    d_I[ei] = I_new;
}
