#include <cuda.h>
#include <cuda_runtime.h>

#ifndef NUMBER_THREADS
#define NUMBER_THREADS 256
#endif

typedef float fp;  // Adjust if fp is defined elsewhere as double; keep consistent

__global__ void srad(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                     int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                     fp *d_dE, fp *d_dW, fp d_q0sqr, fp *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;               // current horizontal block index
    int tx = threadIdx.x;              // current horizontal thread index
    int ei = bx * NUMBER_THREADS + tx; // more threads than actual elements !!!
    int row;                           // column, x position
    int col;                           // row, y position

    // variables
    fp d_Jc;
    fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
    fp d_c_loc;
    fp d_G2, d_L, d_num, d_den, d_qsqr;

    // figure out row/col location in new matrix
    // Use integer arithmetic that avoids expensive modulo where possible
    int idx = ei + 1;
    row = idx % d_Nr - 1;     // (0-n) row
    col = idx / d_Nr;         // (0-n) column
    if (idx % d_Nr == 0) {
        row = d_Nr - 1;
        col = col - 1;
    }

    if (ei < d_Ne) { // make sure that only threads matching jobs run

        // load current element
        d_Jc = d_I[ei];

        // cache neighbor index lookups in registers to reduce repeated loads
        int iN_row = d_iN[row];
        int iS_row = d_iS[row];
        int jW_col = d_jW[col];
        int jE_col = d_jE[col];

        int base_col = d_Nr * col;

        // Use __ldg for potentially read-only data (for architectures >= Kepler),
        // but keep standard loads for broad compatibility and simplicity
        fp north = d_I[iN_row + base_col];
        fp south = d_I[iS_row + base_col];
        fp west  = d_I[row + d_Nr * jW_col];
        fp east  = d_I[row + d_Nr * jE_col];

        // directional derivatives
        d_dN_loc = north - d_Jc; // north direction derivative
        d_dS_loc = south - d_Jc; // south direction derivative
        d_dW_loc = west  - d_Jc; // west direction derivative
        d_dE_loc = east  - d_Jc; // east direction derivative

        // normalized discrete gradient mag squared (equ 52,53)
        fp Jc2 = d_Jc * d_Jc;
        fp g2_num = d_dN_loc * d_dN_loc + d_dS_loc * d_dS_loc +
                    d_dW_loc * d_dW_loc + d_dE_loc * d_dE_loc;
        d_G2 = g2_num / Jc2;

        // normalized discrete laplacian (equ 54)
        fp lap_num = d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc;
        d_L = lap_num / d_Jc;

        // ICOV (equ 31/35)
        fp L2 = d_L * d_L;
        d_num = fmaf(-1.0f / 16.0f, L2, 0.5f * d_G2); // (0.5*G2) - (1/16)*L^2
        d_den = fmaf(0.25f, d_L, 1.0f);               // 1 + 0.25*L
        fp den2 = d_den * d_den;
        d_qsqr = d_num / den2;

        // diffusion coefficient (equ 33)
        fp q0_term = d_q0sqr * (1.0f + d_q0sqr);
        d_den = (d_qsqr - d_q0sqr) / q0_term;
        d_c_loc = 1.0f / (1.0f + d_den);

        // saturate diffusion coefficient to 0-1 range using branchless clamp
        d_c_loc = fminf(fmaxf(d_c_loc, 0.0f), 1.0f);

        // store to global memory (kept separate to favor coalescing)
        d_dN[ei] = d_dN_loc;
        d_dS[ei] = d_dS_loc;
        d_dW[ei] = d_dW_loc;
        d_dE[ei] = d_dE_loc;
        d_c[ei]  = d_c_loc;
    }
}
