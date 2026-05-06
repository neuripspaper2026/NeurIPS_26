#include <cuda_runtime.h>

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
    int idxp1 = ei + 1;
    row = idxp1 - (idxp1 / d_Nr) * d_Nr - 1; // (0-n) row, optimized modulo
    col = idxp1 / d_Nr;                      // (0-n) column
    if (idxp1 % d_Nr == 0) {
        row = d_Nr - 1;
        col = col - 1;
    }

    if (ei < d_Ne) { // make sure that only threads matching jobs run

        // directional derivatives, ICOV, diffusion coefficent
        d_Jc = d_I[ei]; // get value of the current element

        // cache indices locally to help compiler keep them in registers
        int row_iN = d_iN[row];
        int row_iS = d_iS[row];
        int col_jW = d_jW[col];
        int col_jE = d_jE[col];

        int base_col = d_Nr * col;

        // directional derivates (every element of IMAGE)
        d_dN_loc = d_I[row_iN + base_col] - d_Jc; // north direction derivative
        d_dS_loc = d_I[row_iS + base_col] - d_Jc; // south direction derivative
        d_dW_loc =
            d_I[row + d_Nr * col_jW] - d_Jc; // west direction derivative
        d_dE_loc =
            d_I[row + d_Nr * col_jE] - d_Jc; // east direction derivative

        // normalized discrete gradient mag squared (equ 52,53)
        fp d_dN_sq = d_dN_loc * d_dN_loc;
        fp d_dS_sq = d_dS_loc * d_dS_loc;
        fp d_dW_sq = d_dW_loc * d_dW_loc;
        fp d_dE_sq = d_dE_loc * d_dE_loc;

        fp Jc_sq = d_Jc * d_Jc;

        d_G2 = (d_dN_sq + d_dS_sq + d_dW_sq + d_dE_sq) /
               Jc_sq; // gradient (based on derivatives)

        // normalized discrete laplacian (equ 54)
        fp sum_d = d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc;
        d_L = sum_d / d_Jc; // laplacian (based on derivatives)

        // ICOV (equ 31/35)
        d_num = (fp)0.5 * d_G2 -
                ((fp)(1.0 / 16.0) *
                 (d_L * d_L));        // num (based on gradient and laplacian)
        d_den = (fp)1.0 + (fp)0.25 * d_L; // den (based on laplacian)
        fp d_den_sq = d_den * d_den;
        d_qsqr = d_num / d_den_sq; // qsqr (based on num and den)

        // diffusion coefficent (equ 33) (every element of IMAGE)
        fp q0_term = d_q0sqr * ((fp)1.0 + d_q0sqr);
        d_den = (d_qsqr - d_q0sqr) / q0_term; // den (based on qsqr and q0sqr)
        d_c_loc = (fp)1.0 / ((fp)1.0 + d_den); // diffusion coefficient

        // saturate diffusion coefficent to 0-1 range using branchless clamp
        d_c_loc = fminf(fmaxf(d_c_loc, (fp)0.0), (fp)1.0);

        // save data to global memory
        d_dN[ei] = d_dN_loc;
        d_dS[ei] = d_dS_loc;
        d_dW[ei] = d_dW_loc;
        d_dE[ei] = d_dE_loc;
        d_c[ei] = d_c_loc;
    }
}
