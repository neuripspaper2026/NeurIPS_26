#include <cuda_runtime.h>

__global__ void srad2(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                      int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                      fp *d_dE, fp *d_dW, fp *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;               // get current horizontal block index
    int tx = threadIdx.x;              // get current horizontal thread index
    int ei = bx * NUMBER_THREADS + tx; // more threads than actual elements !!!
    int row;                           // column, x position
    int col;                           // row, y position

    // variables
    fp d_cN, d_cS, d_cW, d_cE;
    fp d_D;

    // figure out row/col location in new matrix
    int idxp1 = ei + 1;
    row = idxp1 - (idxp1 / d_Nr) * d_Nr - 1; // (0-n) row, optimized modulo
    col = idxp1 / d_Nr;                      // (0-n) column
    if (idxp1 % d_Nr == 0) {
        row = d_Nr - 1;
        col = col - 1;
    }

    if (ei < d_Ne) { // make sure that only threads matching jobs run

        // cache index lookups locally to reduce redundant global memory reads
        int row_iS = d_iS[row];
        int col_jE = d_jE[col];
        int base_col = d_Nr * col;

        // diffusion coefficent
        d_cN = d_c[ei];                       // north diffusion coefficient
        d_cS = d_c[row_iS + base_col];        // south diffusion coefficient
        d_cW = d_c[ei];                       // west diffusion coefficient
        d_cE = d_c[row + d_Nr * col_jE];      // east diffusion coefficient

        // divergence (equ 58)
        fp d_dN_loc = d_dN[ei];
        fp d_dS_loc = d_dS[ei];
        fp d_dW_loc = d_dW[ei];
        fp d_dE_loc = d_dE[ei];

        d_D = d_cN * d_dN_loc + d_cS * d_dS_loc + d_cW * d_dW_loc +
              d_cE * d_dE_loc; // divergence

        // image update (equ 61) (every element of IMAGE)
        fp Iei = d_I[ei];
        d_I[ei] = Iei + (fp)0.25 * d_lambda * d_D;
    }
}
