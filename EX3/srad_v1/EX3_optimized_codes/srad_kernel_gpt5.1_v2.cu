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
    int bx = blockIdx.x;               // current horizontal block index
    int tx = threadIdx.x;              // current horizontal thread index
    int ei = bx * NUMBER_THREADS + tx; // element index (may exceed d_Ne)
    int row;                           // row position
    int col;                           // column position

    // variables
    fp d_Jc = 0.0f;
    fp d_dN_loc = 0.0f, d_dS_loc = 0.0f, d_dW_loc = 0.0f, d_dE_loc = 0.0f;
    fp d_c_loc = 0.0f;
    fp d_G2 = 0.0f, d_L = 0.0f, d_num = 0.0f, d_den = 0.0f, d_qsqr = 0.0f;

    // Precompute some constants using single precision and fast-math friendly forms
    const fp c_0_5  = 0.5f;
    const fp c_1_16 = 0.0625f;   // 1.0 / 16.0
    const fp c_0_25 = 0.25f;

    // figure out row/col location in new matrix
    int t = ei + 1;
    row = t % d_Nr - 1;
    col = t / d_Nr;
    if (t % d_Nr == 0) {
        row = d_Nr - 1;
        col = col - 1;
    }

    if (ei < d_Ne) {

        // load center value
        d_Jc = d_I[ei];

        // directional derivatives (use local variables for indices to encourage reuse)
        const int idxN = d_iN[row] + d_Nr * col;
        const int idxS = d_iS[row] + d_Nr * col;
        const int idxW = row + d_Nr * d_jW[col];
        const int idxE = row + d_Nr * d_jE[col];

        d_dN_loc = d_I[idxN] - d_Jc;
        d_dS_loc = d_I[idxS] - d_Jc;
        d_dW_loc = d_I[idxW] - d_Jc;
        d_dE_loc = d_I[idxE] - d_Jc;

        // normalized discrete gradient mag squared (equ 52,53)
        fp sum_sq = d_dN_loc * d_dN_loc + d_dS_loc * d_dS_loc +
                    d_dW_loc * d_dW_loc + d_dE_loc * d_dE_loc;
        fp inv_Jc2 = 1.0f / (d_Jc * d_Jc);
        d_G2 = sum_sq * inv_Jc2;

        // normalized discrete laplacian (equ 54)
        fp sum_dir = d_dN_loc + d_dS_loc + d_dW_loc + d_dE_loc;
        fp inv_Jc = 1.0f / d_Jc;
        d_L = sum_dir * inv_Jc;

        // ICOV (equ 31/35)
        d_num = c_0_5 * d_G2 - c_1_16 * d_L * d_L;
        d_den = 1.0f + c_0_25 * d_L;
        fp inv_den = 1.0f / (d_den * d_den);
        d_qsqr = d_num * inv_den;

        // diffusion coefficient (equ 33)
        fp denom_q = d_q0sqr * (1.0f + d_q0sqr);
        d_den = (d_qsqr - d_q0sqr) / denom_q;
        d_c_loc = 1.0f / (1.0f + d_den);

        // saturate diffusion coefficent to 0-1 range
        d_c_loc = fminf(fmaxf(d_c_loc, 0.0f), 1.0f);

        // save data to global memory
        d_dN[ei] = d_dN_loc;
        d_dS[ei] = d_dS_loc;
        d_dW[ei] = d_dW_loc;
        d_dE[ei] = d_dE_loc;
        d_c[ei]  = d_c_loc;
    }
}
