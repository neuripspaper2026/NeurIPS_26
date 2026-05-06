#include <cuda.h>
#include <cuda_runtime.h>

#define NUMBER_THREADS 256

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
    // use integer arithmetic to avoid expensive mod/div when possible
    int ei1 = ei + 1;
    col = ei1 / d_Nr;
    row = ei1 - col * d_Nr - 1;
    if (ei1 % d_Nr == 0) {
        row = d_Nr - 1;
        col = col - 1;
    }

    if (ei < d_Ne) { // make sure that only threads matching jobs run

        // directional derivatives, ICOV, diffusion coefficent
        d_Jc = d_I[ei]; // get value of the current element

        // precompute commonly used products
        int colNr = d_Nr * col;
        int idxN = d_iN[row] + colNr;
        int idxS = d_iS[row] + colNr;
        int idxW = row + d_Nr * d_jW[col];
        int idxE = row + d_Nr * d_jE[col];

        // directional derivates (every element of IMAGE)
        d_dN_loc = d_I[idxN] - d_Jc; // north direction derivative
        d_dS_loc = d_I[idxS] - d_Jc; // south direction derivative
        d_dW_loc = d_I[idxW] - d_Jc; // west direction derivative
        d_dE_loc = d_I[idxE] - d_Jc; // east direction derivative

        // normalized discrete gradient mag squared (equ 52,53)
        fp d_Jc_sq = d_Jc * d_Jc;
        fp grad_sq = d_dN_loc * d_dN_loc + d_dS_loc * d_dS_loc +
                     d_dW_loc * d_dW_loc + d_dE_loc * d_dE_loc;
        d_G2 = grad_sq / d_Jc_sq; // gradient (based on derivatives)

        // normalized discrete laplacian (equ 54)
        fp lap = d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc;
        d_L = lap / d_Jc; // laplacian (based on derivatives)

        // ICOV (equ 31/35)
        fp L_sq = d_L * d_L;
        d_num = __fmaf_rn(-1.0f / 16.0f, L_sq, 0.5f * d_G2);
        d_den = __fmaf_rn(0.25f, d_L, 1.0f);
        fp d_den_sq = d_den * d_den;
        d_qsqr = d_num / d_den_sq; // qsqr (based on num and den)

        // diffusion coefficent (equ 33) (every element of IMAGE)
        fp q0_term = d_q0sqr * (1.0f + d_q0sqr);
        d_den = (d_qsqr - d_q0sqr) / q0_term;
        d_c_loc = 1.0f / (1.0f + d_den); // diffusion coefficient (based on den)

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
