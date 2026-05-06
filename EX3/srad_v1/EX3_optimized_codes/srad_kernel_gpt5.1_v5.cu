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
    row = (ei + 1) % d_Nr - 1;     // (0-n) row
    col = (ei + 1) / d_Nr + 1 - 1; // (0-n) column
    if ((ei + 1) % d_Nr == 0) {
        row = d_Nr - 1;
        col = col - 1;
    }

    if (ei < d_Ne) { // make sure that only threads matching jobs run

        // directional derivatives, ICOV, diffusion coefficent
        d_Jc = d_I[ei]; // get value of the current element

        // directional derivates (every element of IMAGE)(try to copy to shared
        // memory or temp files)
        int row_offset = d_Nr * col;

        fp n_val = d_I[d_iN[row] + row_offset];
        fp s_val = d_I[d_iS[row] + row_offset];
        fp w_val = d_I[row + d_Nr * d_jW[col]];
        fp e_val = d_I[row + d_Nr * d_jE[col]];

        d_dN_loc = n_val - d_Jc; // north direction derivative
        d_dS_loc = s_val - d_Jc; // south direction derivative
        d_dW_loc = w_val - d_Jc; // west direction derivative
        d_dE_loc = e_val - d_Jc; // east direction derivative

        // normalized discrete gradient mag squared (equ 52,53)
        fp d_Jc_sq = d_Jc * d_Jc;
        fp g2_num = d_dN_loc * d_dN_loc + d_dS_loc * d_dS_loc +
                    d_dW_loc * d_dW_loc + d_dE_loc * d_dE_loc;
        d_G2 = g2_num / d_Jc_sq; // gradient (based on derivatives)

        // normalized discrete laplacian (equ 54)
        fp l_num = d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc;
        d_L = l_num / d_Jc; // laplacian (based on derivatives)

        // ICOV (equ 31/35)
        fp d_L_sq = d_L * d_L;
        d_num = __fmaf_rn(-0.0625f, d_L_sq, 0.5f * d_G2); // 0.5*G2 - (1/16)*L^2
        d_den = __fmaf_rn(0.25f, d_L, 1.0f);              // 1 + 0.25*L
        fp d_den_sq = d_den * d_den;
        d_qsqr = d_num / d_den_sq; // qsqr (based on num and den)

        // diffusion coefficent (equ 33) (every element of IMAGE)
        fp denom_q = d_q0sqr * (1.0f + d_q0sqr);
        d_den = (d_qsqr - d_q0sqr) / denom_q; // den (based on qsqr and q0sqr)
        d_c_loc = 1.0f / (1.0f + d_den);      // diffusion coefficient (based on den)

        // saturate diffusion coefficent to 0-1 range
        d_c_loc = fminf(fmaxf(d_c_loc, 0.0f), 1.0f);

        // save data to global memory
        d_dN[ei] = d_dN_loc;
        d_dS[ei] = d_dS_loc;
        d_dW[ei] = d_dW_loc;
        d_dE[ei] = d_dE_loc;
        d_c[ei] = d_c_loc;
    }
}
