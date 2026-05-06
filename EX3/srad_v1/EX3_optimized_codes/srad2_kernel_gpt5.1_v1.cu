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
    int bx = blockIdx.x;               // get current horizontal block index
    int tx = threadIdx.x;              // get current horizontal thread index
    int ei = bx * blockDim.x + tx;     // use actual blockDim.x for flexibility
    int row;                           // column, x position
    int col;                           // row, y position

    // variables
    fp d_cN, d_cS, d_cW, d_cE;
    fp d_D;

    if (ei < d_Ne) { // make sure that only threads matching jobs run

        // figure out row/col location in new matrix (convert global index to 2D)
        row = ei % d_Nr;   // 0 .. d_Nr-1
        col = ei / d_Nr;   // 0 .. d_Nc-1

        // diffusion coefficent
        const int iS_row = d_iS[row];
        const int jE_col = d_jE[col];

        const int idx_self = ei;
        const int idx_south = iS_row + d_Nr * col;
        const int idx_east  = row + d_Nr * jE_col;

        d_cN = d_c[idx_self];   // north diffusion coefficient (same as center)
        d_cW = d_c[idx_self];   // west diffusion coefficient (same as center)
        d_cS = d_c[idx_south];  // south diffusion coefficient
        d_cE = d_c[idx_east];   // east diffusion coefficient

        // divergence (equ 58)
        const fp d_dN_loc = d_dN[idx_self];
        const fp d_dS_loc = d_dS[idx_self];
        const fp d_dW_loc = d_dW[idx_self];
        const fp d_dE_loc = d_dE[idx_self];

        d_D = d_cN * d_dN_loc + d_cS * d_dS_loc +
              d_cW * d_dW_loc + d_cE * d_dE_loc; // divergence

        // image update (equ 61) (every element of IMAGE)
        // use local register for I to reduce global traffic and rely on FMA
        fp I_val = d_I[idx_self];
        I_val = fmaf(fp(0.25) * d_lambda, d_D, I_val);
        d_I[idx_self] = I_val;
    }
}
