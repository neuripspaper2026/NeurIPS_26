#include <cuda.h>
#include <cuda_runtime.h>

#ifndef NUMBER_THREADS
#define NUMBER_THREADS 256
#endif

typedef float fp;  // Keep consistent with srad_kernel.cu

__global__ void srad2(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                      int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                      fp *d_dE, fp *d_dW, fp *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;               // current horizontal block index
    int tx = threadIdx.x;              // current horizontal thread index
    int ei = bx * NUMBER_THREADS + tx; // linear element index
    int row;                           // row (y position)
    int col;                           // column (x position)

    // variables
    fp d_cN, d_cS, d_cW, d_cE;
    fp d_D;

    // figure out row/col location in new matrix (consistent with srad_kernel.cu)
    int idx = ei + 1;
    row = idx % d_Nr - 1; // (0-n) row
    col = idx / d_Nr;     // (0-n) column
    if (idx % d_Nr == 0) {
        row = d_Nr - 1;
        col = col - 1;
    }

    if (ei < d_Ne) { // ensure only valid elements are processed

        // precompute column base index for better ILP and fewer multiplies
        int base_col = d_Nr * col;

        // load diffusion coefficients; cache neighbor index lookups in registers
        d_cN = d_c[ei];                          // north diffusion coefficient
        int iS_row = d_iS[row];
        d_cS = d_c[iS_row + base_col];           // south diffusion coefficient
        d_cW = d_c[ei];                          // west diffusion coefficient
        int jE_col = d_jE[col];
        d_cE = d_c[row + d_Nr * jE_col];         // east diffusion coefficient

        // divergence (equ 58)
        fp d_dN_loc = d_dN[ei];
        fp d_dS_loc = d_dS[ei];
        fp d_dW_loc = d_dW[ei];
        fp d_dE_loc = d_dE[ei];

        d_D = fmaf(d_cN, d_dN_loc,
              fmaf(d_cS, d_dS_loc,
              fmaf(d_cW, d_dW_loc, d_cE * d_dE_loc)));

        // image update (equ 61) (every element of IMAGE)
        // use FMA to combine multiply-add for better throughput/precision
        fp I_old = d_I[ei];
        fp scale = 0.25f * d_lambda;
        d_I[ei] = fmaf(scale, d_D, I_old);
    }
}
