__global__ void srad2(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                      int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                      fp *d_dE, fp *d_dW, fp *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ei = bx * NUMBER_THREADS + tx;

    if (ei >= d_Ne) return;

    // Calculate row/col using optimized integer arithmetic (consistent with srad_kernel.cu)
    int row = ei % d_Nr;
    int col = ei / d_Nr;

    // Precompute common index calculations
    int row_iS = d_iS[row];
    int col_jE = d_jE[col];
    
    int base_col = d_Nr * col;
    
    // Load diffusion coefficients with coalesced memory access pattern
    fp d_cN = d_c[ei];
    fp d_cS = d_c[row_iS + base_col];
    fp d_cW = d_c[ei];
    fp d_cE = d_c[row + d_Nr * col_jE];

    // Load directional derivatives (coalesced access)
    fp dN = d_dN[ei];
    fp dS = d_dS[ei];
    fp dW = d_dW[ei];
    fp dE = d_dE[ei];

    // Compute divergence (equ 58) using FMA operations
    fp d_D = fmaf(d_cN, dN, fmaf(d_cS, dS, fmaf(d_cW, dW, d_cE * dE)));

    // Load current image value
    fp I_current = d_I[ei];

    // Image update (equ 61) using FMA for better precision and performance
    fp update = fmaf(fp(0.25) * d_lambda, d_D, I_current);

    // Coalesced write to global memory
    d_I[ei] = update;
}
