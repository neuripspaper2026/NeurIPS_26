__global__ void srad(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                     int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                     fp *d_dE, fp *d_dW, fp d_q0sqr, fp *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;               // get current horizontal block index
    int tx = threadIdx.x;              // get current horizontal thread index
    int ei = bx * blockDim.x + tx;     // more threads than actual elements !!!
    
    // Early exit if thread index exceeds number of elements
    if (ei >= d_Ne) return;

    // Shared memory for caching frequently accessed data
    extern __shared__ fp shared_I[];

    // Figure out row/col location in new matrix
    int row = ei % d_Nr;
    int col = ei / d_Nr;

    // Variables
    fp d_Jc;
    fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
    fp d_c_loc;
    fp d_G2, d_L, d_num, d_den, d_qsqr;

    // Load data into shared memory cooperatively
    // Each thread loads one element
    shared_I[tx] = d_I[ei];
    __syncthreads();

    // Get value of the current element
    d_Jc = shared_I[tx];

    // Directional derivatives with bounds checking
    int idx_N = d_iN[row] + d_Nr * col;
    int idx_S = d_iS[row] + d_Nr * col;
    int idx_W = row + d_Nr * d_jW[col];
    int idx_E = row + d_Nr * d_jE[col];

    // Load neighbor values from global memory
    fp I_N = d_I[idx_N];
    fp I_S = d_I[idx_S];
    fp I_W = d_I[idx_W];
    fp I_E = d_I[idx_E];

    // Compute directional derivatives
    d_dN_loc = I_N - d_Jc;
    d_dS_loc = I_S - d_Jc;
    d_dW_loc = I_W - d_Jc;
    d_dE_loc = I_E - d_Jc;

    // Normalized discrete gradient mag squared (equ 52,53)
    d_G2 = (d_dN_loc * d_dN_loc + d_dS_loc * d_dS_loc +
            d_dW_loc * d_dW_loc + d_dE_loc * d_dE_loc) /
           (d_Jc * d_Jc);

    // Normalized discrete laplacian (equ 54)
    d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;

    // ICOV (equ 31/35)
    d_num = (0.5f * d_G2) - ((1.0f / 16.0f) * (d_L * d_L));
    d_den = 1.0f + (0.25f * d_L);
    d_qsqr = d_num / (d_den * d_den);

    // Diffusion coefficient (equ 33)
    d_den = (d_qsqr - d_q0sqr) / (d_q0sqr * (1.0f + d_q0sqr));
    d_c_loc = 1.0f / (1.0f + d_den);

    // Saturate diffusion coefficient to 0-1 range
    d_c_loc = fmaxf(0.0f, fminf(1.0f, d_c_loc));

    // Save data to global memory
    d_dN[ei] = d_dN_loc;
    d_dS[ei] = d_dS_loc;
    d_dW[ei] = d_dW_loc;
    d_dE[ei] = d_dE_loc;
    d_c[ei] = d_c_loc;
}
