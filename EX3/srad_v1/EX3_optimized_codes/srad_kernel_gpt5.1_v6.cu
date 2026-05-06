#include <cuda.h>
#include <cuda_runtime.h>

#ifndef NUMBER_THREADS
#define NUMBER_THREADS 256
#endif

typedef float fp;

__global__ void srad(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                     int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                     fp *d_dE, fp *d_dW, fp d_q0sqr, fp *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;               // current horizontal block index
    int tx = threadIdx.x;              // current horizontal thread index
    int ei = bx * NUMBER_THREADS + tx; // linear element index
    int row;                           // column, x position
    int col;                           // row, y position

    // variables
    fp d_Jc;
    fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
    fp d_c_loc;
    fp d_G2, d_L, d_num, d_den, d_qsqr;

    // figure out row/col location in matrix (0-based)
    row = ei % d_Nr;
    col = ei / d_Nr;

    if (ei < d_Ne) { // make sure that only threads matching jobs run

        // directional derivatives, ICOV, diffusion coefficient
        d_Jc = d_I[ei]; // current element value

        int iN = d_iN[row];
        int iS = d_iS[row];
        int jW = d_jW[col];
        int jE = d_jE[col];

        // directional derivatives
        d_dN_loc = d_I[iN + d_Nr * col] - d_Jc; // north
        d_dS_loc = d_I[iS + d_Nr * col] - d_Jc; // south
        d_dW_loc = d_I[row + d_Nr * jW] - d_Jc; // west
        d_dE_loc = d_I[row + d_Nr * jE] - d_Jc; // east

        // normalized discrete gradient mag squared (equ 52,53)
        fp d_Jc2 = d_Jc * d_Jc;
        fp g2_acc = d_dN_loc * d_dN_loc + d_dS_loc * d_dS_loc +
                    d_dW_loc * d_dW_loc + d_dE_loc * d_dE_loc;
        d_G2 = g2_acc / d_Jc2; // gradient (based on derivatives)

        // normalized discrete laplacian (equ 54)
        fp lap_acc = d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc;
        d_L = lap_acc / d_Jc; // laplacian (based on derivatives)

        // ICOV (equ 31/35)
        fp L2 = d_L * d_L;
        d_num = 0.5f * d_G2 - (1.0f / 16.0f) * L2; // num (based on gradient and laplacian)
        d_den = 1.0f + 0.25f * d_L;               // den (based on laplacian)
        fp d_den2 = d_den * d_den;
        d_qsqr = d_num / d_den2; // qsqr (based on num and den)

        // diffusion coefficient (equ 33)
        fp q0_term = d_q0sqr * (1.0f + d_q0sqr);
        d_den = (d_qsqr - d_q0sqr) / q0_term;
        d_c_loc = 1.0f / (1.0f + d_den);

        // saturate diffusion coefficient to 0-1 range using clamp
        d_c_loc = fminf(fmaxf(d_c_loc, 0.0f), 1.0f);

        // save data to global memory
        d_dN[ei] = d_dN_loc;
        d_dS[ei] = d_dS_loc;
        d_dW[ei] = d_dW_loc;
        d_dE[ei] = d_dE_loc;
        d_c[ei] = d_c_loc;
    }
}
