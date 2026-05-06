__global__ void srad2(fp d_lambda, int d_Nr, int d_Nc, long d_Ne, int *d_iN,
                      int *d_iS, int *d_jE, int *d_jW, fp *d_dN, fp *d_dS,
                      fp *d_dE, fp *d_dW, fp *d_c, fp *d_I) {

    // indexes
    int bx = blockIdx.x;               // get current horizontal block index
    int tx = threadIdx.x;              // get current horizontal thread index
    int ei = bx * blockDim.x + tx;     // more threads than actual elements !!!

    // Early exit if thread index exceeds number of elements
    if (ei >= d_Ne) return;

    // Shared memory for caching frequently accessed data
    extern __shared__ fp shared_c[];

    // Figure out row/col location in new matrix
    int row = ei % d_Nr;
    int col = ei / d_Nr;

    // Variables
    fp d_cN, d_cS, d_cW, d_cE;
    fp d_D;

    // Load diffusion coefficients into shared memory cooperatively
    shared_c[tx] = d_c[ei];
    __syncthreads();

    // Get current diffusion coefficient
    d_cN = shared_c[tx];                     // north diffusion coefficient
    d_cW = shared_c[tx];                     // west diffusion coefficient

    // Load neighbor diffusion coefficients from global memory
    d_cS = d_c[d_iS[row] + d_Nr * col];      // south diffusion coefficient
    d_cE = d_c[row + d_Nr * d_jE[col]];      // east diffusion coefficient

    // Compute divergence (equ 58)
    d_D = d_cN * d_dN[ei] + d_cS * d_dS[ei] + d_cW * d_dW[ei] + d_cE * d_dE[ei]; // divergence

    // Image update (equ 61) (every element of IMAGE)
    d_I[ei] = d_I[ei] + 0.25f * d_lambda * d_D; // updates image (based on input time step and divergence)
}
