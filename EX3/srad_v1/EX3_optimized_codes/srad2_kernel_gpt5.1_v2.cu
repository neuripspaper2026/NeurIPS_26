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
    int ei = bx * NUMBER_THREADS + tx; // element index (may exceed d_Ne)
    int row;                           // row position
    int col;                           // column position

    // variables
    fp d_cN, d_cS, d_cW, d_cE;
    fp d_D;

    // precompute constants
    const fp c_0_25 = 0.25f;

    // figure out row/col location in new matrix
    int t = ei + 1;
    row = t % d_Nr - 1; // (0-n) row
    col = t / d_Nr;     // (0-n) column
    if (t % d_Nr == 0) {
        row = d_Nr - 1;
        col = col - 1;
    }

    if (ei < d_Ne) { // make sure that only threads matching jobs run

        // diffusion coefficient indices
        const int idxN = ei;                          // north diffusion coefficient
        const int idxS = d_iS[row] + d_Nr * col;      // south diffusion coefficient
        const int idxW = ei;                          // west diffusion coefficient
        const int idxE = row + d_Nr * d_jE[col];      // east diffusion coefficient

        // load diffusion coefficients from global memory
        d_cN = d_c[idxN];
        d_cS = d_c[idxS];
        d_cW = d_c[idxW];
        d_cE = d_c[idxE];

        // divergence (equ 58)
        fp d_dN_loc = d_dN[ei];
        fp d_dS_loc = d_dS[ei];
        fp d_dW_loc = d_dW[ei];
        fp d_dE_loc = d_dE[ei];

        d_D = d_cN * d_dN_loc +
              d_cS * d_dS_loc +
              d_cW * d_dW_loc +
              d_cE * d_dE_loc; // divergence

        // image update (equ 61) (every element of IMAGE)
        fp I_old = d_I[ei];
        d_I[ei] = I_old + c_0_25 * d_lambda * d_D;
    }
}
