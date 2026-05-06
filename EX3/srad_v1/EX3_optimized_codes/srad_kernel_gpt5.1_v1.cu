#include <cuda.h>
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
    int ei = bx * blockDim.x + tx;     // use actual blockDim.x for flexibility
    int row;                           // column, x position
    int col;                           // row, y position

    // variables
    fp d_Jc;
    fp d_dN_loc, d_dS_loc, d_dW_loc, d_dE_loc;
    fp d_c_loc;
    fp d_G2, d_L, d_num, d_den, d_qsqr;

    if (ei < d_Ne) {

        // figure out row/col location in new matrix (convert global index to 2D)
        // Using integer division/modulo with precomputed Nr to avoid recomputing (ei+1)
        row = ei % d_Nr;   // 0 .. d_Nr-1
        col = ei / d_Nr;   // 0 .. d_Nc-1

        // directional derivatives, ICOV, diffusion coefficent
        d_Jc = d_I[ei]; // get value of the current element

        // directional derivates (every element of IMAGE)
        const int iN_row = d_iN[row];
        const int iS_row = d_iS[row];
        const int jW_col = d_jW[col];
        const int jE_col = d_jE[col];

        const int north_idx = iN_row + d_Nr * col;
        const int south_idx = iS_row + d_Nr * col;
        const int west_idx  = row + d_Nr * jW_col;
        const int east_idx  = row + d_Nr * jE_col;

        d_dN_loc = d_I[north_idx] - d_Jc; // north direction derivative
        d_dS_loc = d_I[south_idx] - d_Jc; // south direction derivative
        d_dW_loc = d_I[west_idx]  - d_Jc; // west direction derivative
        d_dE_loc = d_I[east_idx]  - d_Jc; // east direction derivative

        // normalized discrete gradient mag squared (equ 52,53)
        const fp d_Jc2 = d_Jc * d_Jc;
        d_G2 = (d_dN_loc * d_dN_loc + d_dS_loc * d_dS_loc +
                d_dW_loc * d_dW_loc + d_dE_loc * d_dE_loc) / d_Jc2;

        // normalized discrete laplacian (equ 54)
        d_L = (d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc) / d_Jc;

        // ICOV (equ 31/35)
        d_num = fp(0.5) * d_G2 - fp(1.0 / 16.0) * (d_L * d_L);
        d_den = fp(1.0) + fp(0.25) * d_L;
        d_qsqr = d_num / (d_den * d_den);

        // diffusion coefficent (equ 33) (every element of IMAGE)
        const fp one = fp(1.0);
        d_den = (d_qsqr - d_q0sqr) / (d_q0sqr * (one + d_q0sqr));
        d_c_loc = one / (one + d_den);

        // saturate diffusion coefficent to 0-1 range
        // use fminf/fmaxf for branchless clamping
        d_c_loc = fminf(fmaxf(d_c_loc, fp(0.0)), one);

        // save data to global memory
        d_dN[ei] = d_dN_loc;
        d_dS[ei] = d_dS_loc;
        d_dW[ei] = d_dW_loc;
        d_dE[ei] = d_dE_loc;
        d_c[ei]  = d_c_loc;
    }
}
