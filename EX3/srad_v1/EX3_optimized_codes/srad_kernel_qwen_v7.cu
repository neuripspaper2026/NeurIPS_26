__global__ void srad(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                     int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                     fp *d_dE, fp *d_dW, fp d_q0sqr, fp *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;               // get current horizontal block index
    int tx = threadIdx.x;              // get current horizontal thread index
    int ei = bx * blockDim.x + tx;     // more threads than actual elements !!!
    
    // Early exit for out-of-bounds threads
    if (ei >= d_Ne) return;

    // variables
    fp d_Jc;
    fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
    fp d_c_loc;
    fp d_G2, d_L, d_num, d_den, d_qsqr;

    // figure out row/col location in new matrix
    int row = ei % d_Nr;     // (0-n) row
    int col = ei / d_Nr;     // (0-n) column

    // Load d_I[ei] once into register
    d_Jc = d_I[ei];

    // Load neighbor indices into registers to reduce memory accesses
    int iN_val = d_iN[row];
    int iS_val = d_iS[row];
    int jW_val = d_jW[col];
    int jE_val = d_jE[col];

    // directional derivatives
    d_dN_loc = d_I[iN_val + d_Nr * col] - d_Jc; // north direction derivative
    d_dS_loc = d_I[iS_val + d_Nr * col] - d_Jc; // south direction derivative
    d_dW_loc = d_I[row + d_Nr * jW_val] - d_Jc; // west direction derivative
    d_dE_loc = d_I[row + d_Nr * jE_val] - d_Jc; // east direction derivative

    // normalized discrete gradient mag squared (equ 52,53)
    fp jc_squared = d_Jc * d_Jc;
    d_G2 = (d_dN_loc * d_dN_loc + d_dS_loc * d_dS_loc +
            d_dW_loc * d_dW_loc + d_dE_loc * d_dE_loc) /
           jc_squared;

    // normalized discrete laplacian (equ 54)
    d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;

    // ICOV (equ 31/35)
    d_num = (0.5f * d_G2) - ((1.0f / 16.0f) * (d_L * d_L));
    d_den = 1.0f + (0.25f * d_L);
    d_qsqr = d_num / (d_den * d_den);

    // diffusion coefficient (equ 33)
    d_den = (d_qsqr - d_q0sqr) / (d_q0sqr * (1.0f + d_q0sqr));
    d_c_loc = 1.0f / (1.0f + d_den);

    // saturate diffusion coefficient to 0-1 range
    d_c_loc = fmaxf(0.0f, fminf(1.0f, d_c_loc));

    // save data to global memory
    d_dN[ei] = d_dN_loc;
    d_dS[ei] = d_dS_loc;
    d_dW[ei] = d_dW_loc;
    d_dE[ei] = d_dE_loc;
    d_c[ei] = d_c_loc;
}
