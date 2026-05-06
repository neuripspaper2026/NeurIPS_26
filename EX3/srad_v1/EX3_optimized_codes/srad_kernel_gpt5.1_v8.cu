#include <cuda_runtime.h>

#ifndef NUMBER_THREADS
#define NUMBER_THREADS 256
#endif

typedef float fp;

__global__ void srad(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                     int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                     fp *d_dE, fp *d_dW, fp d_q0sqr, fp *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;               // get current horizontal block index
    int tx = threadIdx.x;              // get current horizontal thread index
    int ei = bx * NUMBER_THREADS + tx; // more threads than actual elements !!!
    int row;                           // column, x position
    int col;                           // row, y position

    // variables
    fp d_Jc;
    fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
    fp d_c_loc;
    fp d_G2, d_L, d_num, d_den, d_qsqr;

    // figure out row/col location in new matrix
    // use integer arithmetic and avoid modulus/divide when possible
    int idx = ei + 1;
    row = idx - (idx / d_Nr) * d_Nr - 1;  // equivalent to (idx % d_Nr) - 1
    col = idx / d_Nr;                     // equivalent to (idx + d_Nr - 1) / d_Nr - 1
    if (idx % d_Nr == 0) {
        row = d_Nr - 1;
        col = col - 1;
    }

    if (ei < d_Ne) { // make sure that only threads matching jobs run

        // directional derivatives, ICOV, diffusion coefficent
        d_Jc = d_I[ei]; // get value of the current element

        int base_idx = d_Nr * col;
        int north_idx = d_iN[row] + base_idx;
        int south_idx = d_iS[row] + base_idx;
        int west_idx  = row + d_Nr * d_jW[col];
        int east_idx  = row + d_Nr * d_jE[col];

        // directional derivates (every element of IMAGE)
        d_dN_loc = d_I[north_idx] - d_Jc; // north direction derivative
        d_dS_loc = d_I[south_idx] - d_Jc; // south direction derivative
        d_dW_loc = d_I[west_idx]  - d_Jc; // west direction derivative
        d_dE_loc = d_I[east_idx]  - d_Jc; // east direction derivative

        // normalized discrete gradient mag squared (equ 52,53)
        fp d_dN_sq = d_dN_loc * d_dN_loc;
        fp d_dS_sq = d_dS_loc * d_dS_loc;
        fp d_dW_sq = d_dW_loc * d_dW_loc;
        fp d_dE_sq = d_dE_loc * d_dE_loc;

        fp JJ = d_Jc * d_Jc;
        d_G2 = (d_dN_sq + d_dS_sq + d_dW_sq + d_dE_sq) / JJ; // gradient

        // normalized discrete laplacian (equ 54)
        d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc; // laplacian

        // ICOV (equ 31/35)
        d_num = fmaf(-0.0625f, d_L * d_L, 0.5f * d_G2); // 0.5*G2 - (1/16)*L^2
        d_den = fmaf(0.25f, d_L, 1.0f);                 // 1 + 0.25*L
        fp d_den_sq = d_den * d_den;
        d_qsqr = d_num / d_den_sq;                      // qsqr

        // diffusion coefficient (equ 33)
        fp q0_term = d_q0sqr * (1.0f + d_q0sqr);
        d_den = (d_qsqr - d_q0sqr) / q0_term;
        d_c_loc = 1.0f / (1.0f + d_den);

        // saturate diffusion coefficent to 0-1 range using min/max to avoid branches
        d_c_loc = fminf(fmaxf(d_c_loc, 0.0f), 1.0f);

        // save data to global memory
        d_dN[ei] = d_dN_loc;
        d_dS[ei] = d_dS_loc;
        d_dW[ei] = d_dW_loc;
        d_dE[ei] = d_dE_loc;
        d_c[ei]  = d_c_loc;
    }
}
