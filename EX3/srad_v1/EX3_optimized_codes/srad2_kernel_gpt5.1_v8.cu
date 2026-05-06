#include <cuda_runtime.h>

#ifndef NUMBER_THREADS
#define NUMBER_THREADS 256
#endif

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
    int idx = ei + 1;
    // compute row/col using integer arithmetic; avoid expensive division/modulo in common case
    int col_tmp = idx / d_Nr;
    int row_tmp = idx - col_tmp * d_Nr; // equivalent to idx % d_Nr

    if (row_tmp != 0) {
        row = row_tmp - 1;
        col = col_tmp;
    } else {
        // handle exact-multiple boundary
        row = d_Nr - 1;
        col = col_tmp - 1;
    }

    if (ei < d_Ne) { // make sure that only threads matching jobs run

        int base_idx = d_Nr * col;
        int south_idx = d_iS[row] + base_idx;
        int east_idx  = row + d_Nr * d_jE[col];

        // diffusion coefficent
        d_cN = d_c[ei];        // north diffusion coefficient (current)
        d_cS = d_c[south_idx]; // south diffusion coefficient
        d_cW = d_cN;           // west diffusion coefficient (same as current)
        d_cE = d_c[east_idx];  // east diffusion coefficient

        // divergence (equ 58)
        fp d_dN_loc = d_dN[ei];
        fp d_dS_loc = d_dS[ei];
        fp d_dW_loc = d_dW[ei];
        fp d_dE_loc = d_dE[ei];

        d_D = fmaf(d_cN, d_dN_loc,
              fmaf(d_cS, d_dS_loc,
              fmaf(d_cW, d_dW_loc,
                   d_cE * d_dE_loc))); // divergence

        // image update (equ 61) (every element of IMAGE)
        fp update = 0.25f * d_lambda * d_D;
        d_I[ei] = d_I[ei] + update;
    }
}
