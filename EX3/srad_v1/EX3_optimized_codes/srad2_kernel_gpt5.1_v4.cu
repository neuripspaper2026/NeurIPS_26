#include <cuda.h>
#include <cuda_runtime.h>

#define NUMBER_THREADS 256

typedef float fp;

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
    // use integer arithmetic to avoid expensive mod/div when possible
    int ei1 = ei + 1;
    col = ei1 / d_Nr;
    row = ei1 - col * d_Nr - 1;
    if (ei1 % d_Nr == 0) {
        row = d_Nr - 1;
        col = col - 1;
    }

    if (ei < d_Ne) { // make sure that only threads matching jobs run

        // precompute commonly used products and indices
        int colNr = d_Nr * col;
        int idxS = d_iS[row] + colNr;
        int idxE = row + d_Nr * d_jE[col];

        // diffusion coefficent
        d_cN = d_c[ei];      // north diffusion coefficient
        d_cS = d_c[idxS];    // south diffusion coefficient
        d_cW = d_c[ei];      // west diffusion coefficient (same as north here)
        d_cE = d_c[idxE];    // east diffusion coefficient

        // divergence (equ 58)
        fp d_dN_loc = d_dN[ei];
        fp d_dS_loc = d_dS[ei];
        fp d_dW_loc = d_dW[ei];
        fp d_dE_loc = d_dE[ei];

        d_D = __fmaf_rn(d_cN, d_dN_loc,
              __fmaf_rn(d_cS, d_dS_loc,
              __fmaf_rn(d_cW, d_dW_loc, d_cE * d_dE_loc)));

        // image update (equ 61) (every element of IMAGE)
        fp I_prev = d_I[ei];
        d_I[ei] = __fmaf_rn(0.25f * d_lambda, d_D, I_prev);
    }
}
