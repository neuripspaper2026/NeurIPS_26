#include <cuda.h>
#include <cuda_runtime.h>

#ifndef NUMBER_THREADS
#define NUMBER_THREADS 256
#endif

typedef float fp;

__global__ void srad2(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                      int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                      fp *d_dE, fp *d_dW, fp *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;               // current horizontal block index
    int tx = threadIdx.x;              // current horizontal thread index
    int ei = bx * NUMBER_THREADS + tx; // linear element index
    int row;                           // row index (0-based)
    int col;                           // column index (0-based)

    // variables
    fp d_cN, d_cS, d_cW, d_cE;
    fp d_D;

    // figure out row/col location in matrix (0-based)
    row = ei % d_Nr;
    col = ei / d_Nr;

    if (ei < d_Ne) { // make sure that only threads matching jobs run

        // pre-load neighbor indices to registers
        int iS = d_iS[row];
        int jE = d_jE[col];

        // diffusion coefficient
        d_cN = d_c[ei];                     // north diffusion coefficient
        d_cS = d_c[iS + d_Nr * col];        // south diffusion coefficient
        d_cW = d_c[ei];                     // west diffusion coefficient
        d_cE = d_c[row + d_Nr * jE];        // east diffusion coefficient

        // divergence (equ 58)
        fp d_dN_loc = d_dN[ei];
        fp d_dS_loc = d_dS[ei];
        fp d_dW_loc = d_dW[ei];
        fp d_dE_loc = d_dE[ei];

        d_D = d_cN * d_dN_loc + d_cS * d_dS_loc + d_cW * d_dW_loc +
              d_cE * d_dE_loc; // divergence

        // image update (equ 61) (every element of IMAGE)
        d_I[ei] = fmaf(0.25f * d_lambda, d_D, d_I[ei]);
    }
}
