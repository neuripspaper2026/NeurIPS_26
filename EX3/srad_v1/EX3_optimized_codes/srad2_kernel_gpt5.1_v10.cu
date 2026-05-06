#include <cuda.h>
#include <cuda_runtime.h>

#ifndef NUMBER_THREADS
#define NUMBER_THREADS 256
#endif

typedef float fp;  // Must match the definition used project-wide

// Optimized SRAD2 kernel for NVIDIA A100
__global__ void srad2(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                      int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                      fp *d_dE, fp *d_dW, fp *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;               // horizontal block index
    int tx = threadIdx.x;              // horizontal thread index
    int ei = bx * NUMBER_THREADS + tx; // element index

    // variables
    fp d_cN, d_cS, d_cW, d_cE;
    fp d_D;

    if (ei < d_Ne) { // ensure only valid elements are processed

        // ---------------------------------------------------------------------
        // Compute row/col from ei with reduced branching, kept consistent
        // with srad_kernel.cu optimization for coalesced access pattern
        // ---------------------------------------------------------------------
        int ei1 = ei + 1;
        int col = ei1 / d_Nr;                  // 0..d_Nc-1
        int row = ei1 - col * d_Nr - 1;        // remainder - 1

        // Boundary fix using predication instead of a branch
        int is_boundary = (row < 0);
        row += is_boundary * d_Nr;   // if boundary, row = d_Nr - 1
        col -= is_boundary;          // if boundary, col--

        // Precompute column base to avoid repeated multiplications
        int col_base = d_Nr * col;

        // ---------------------------------------------------------------------
        // Load diffusion coefficients using read-only cache for better
        // bandwidth utilization on A100
        // ---------------------------------------------------------------------
        // north and west diffusion coefficient are at current element
        d_cN = __ldg(&d_c[ei]);
        d_cW = d_cN;

        // south diffusion coefficient
        int south_idx = d_iS[row] + col_base;
        d_cS = __ldg(&d_c[south_idx]);

        // east diffusion coefficient
        int east_col = __ldg(&d_jE[col]);
        int east_idx = row + d_Nr * east_col;
        d_cE = __ldg(&d_c[east_idx]);

        // ---------------------------------------------------------------------
        // Divergence (equ 58): use FMA to reduce latency where possible
        // ---------------------------------------------------------------------
        fp t0 = fmaf(d_cN, d_dN[ei], d_cS * d_dS[ei]);
        fp t1 = fmaf(d_cW, d_dW[ei], d_cE * d_dE[ei]);
        d_D = t0 + t1;

        // ---------------------------------------------------------------------
        // Image update (equ 61) using fused multiply-add
        // ---------------------------------------------------------------------
        fp scale = 0.25f * d_lambda;
        d_I[ei] = fmaf(scale, d_D, d_I[ei]);
    }
}
