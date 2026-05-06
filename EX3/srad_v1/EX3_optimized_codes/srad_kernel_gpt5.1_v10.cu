#include <cuda.h>
#include <cuda_runtime.h>

#ifndef NUMBER_THREADS
#define NUMBER_THREADS 256
#endif

typedef float fp;  // Adjust if fp is defined differently in the actual project

// Optimized SRAD kernel for NVIDIA A100
__global__ void srad(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                     int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                     fp *d_dE, fp *d_dW, fp d_q0sqr, fp *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;
    int tx = threadIdx.x;
    int ei = bx * NUMBER_THREADS + tx;

    // variables
    fp d_Jc;
    fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
    fp d_c_loc;
    fp d_G2, d_L, d_num, d_den, d_qsqr;

    // Preload reciprocal of d_q0sqr*(1+d_q0sqr) to reduce divisions
    const fp inv_q0 = 1.0f / (d_q0sqr * (1.0f + d_q0sqr));

    // Only compute for valid elements
    if (ei < d_Ne) {

        // ---------------------------------------------------------------------
        // Compute row/col from ei without divergent branch
        // row = (ei + 1) % d_Nr - 1;
        // col = (ei + 1) / d_Nr + 1 - 1;
        // if ((ei + 1) % d_Nr == 0) { row = d_Nr - 1; col = col - 1; }
        // ---------------------------------------------------------------------
        int ei1 = ei + 1;
        int col = ei1 / d_Nr;    // 0..d_Nc-1
        int row = ei1 - col * d_Nr - 1;  // remainder - 1

        // Handle boundary case with predication instead of branch
        int is_boundary = (row < 0);
        row += is_boundary * d_Nr;   // if boundary, row = d_Nr - 1
        col -= is_boundary;          // if boundary, col--

        // ---------------------------------------------------------------------
        // Coalesced load of central intensity value
        // ---------------------------------------------------------------------
        d_Jc = __ldg(&d_I[ei]);

        // Compute base index for this column to reduce repeated multiplications
        int col_base = d_Nr * col;

        // ---------------------------------------------------------------------
        // Directional derivatives (use read-only cache where beneficial)
        // ---------------------------------------------------------------------
        fp north = __ldg(&d_I[d_iN[row] + col_base]);
        fp south = __ldg(&d_I[d_iS[row] + col_base]);
        fp west  = __ldg(&d_I[row + d_Nr * d_jW[col]]);
        fp east  = __ldg(&d_I[row + d_Nr * d_jE[col]]);

        d_dN_loc = north - d_Jc;
        d_dS_loc = south - d_Jc;
        d_dW_loc = west  - d_Jc;
        d_dE_loc = east  - d_Jc;

        // ---------------------------------------------------------------------
        // Normalized discrete gradient magnitude squared
        // ---------------------------------------------------------------------
        fp d_Jc2 = d_Jc * d_Jc;

        d_G2 = (d_dN_loc * d_dN_loc +
                d_dS_loc * d_dS_loc +
                d_dW_loc * d_dW_loc +
                d_dE_loc * d_dE_loc) / d_Jc2;

        // ---------------------------------------------------------------------
        // Normalized discrete Laplacian
        // ---------------------------------------------------------------------
        d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;

        // ---------------------------------------------------------------------
        // ICOV
        // ---------------------------------------------------------------------
        fp d_L2 = d_L * d_L;
        d_num = 0.5f * d_G2 - (0.0625f * d_L2);   // 1/16 = 0.0625
        d_den = 1.0f + 0.25f * d_L;              // 1/4  = 0.25
        fp d_den2 = d_den * d_den;
        d_qsqr = d_num / d_den2;

        // ---------------------------------------------------------------------
        // Diffusion coefficient
        // ---------------------------------------------------------------------
        d_den = (d_qsqr - d_q0sqr) * inv_q0;
        d_c_loc = 1.0f / (1.0f + d_den);

        // Saturate diffusion coefficient to [0, 1] (branchless)
        d_c_loc = fminf(fmaxf(d_c_loc, 0.0f), 1.0f);

        // ---------------------------------------------------------------------
        // Store results (coalesced writes)
        // ---------------------------------------------------------------------
        d_dN[ei] = d_dN_loc;
        d_dS[ei] = d_dS_loc;
        d_dW[ei] = d_dW_loc;
        d_dE[ei] = d_dE_loc;
        d_c[ei]  = d_c_loc;
    }
}
