__global__ void kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu,
                                box_str *d_box_gpu, FOUR_VECTOR *d_rv_gpu,
                                fp *d_qv_gpu, FOUR_VECTOR *d_fv_gpu) {

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
    //	THREAD PARAMETERS
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

    int bx = blockIdx.x;  // get current horizontal block index (0-n)
    int tx = threadIdx.x; // get current horizontal thread index (0-n)
    int lane_id = tx & 31; // lane id within warp (0-31)

    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
    //	DO FOR THE NUMBER OF BOXES
    //--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

    if (bx < d_dim_gpu.number_boxes) {

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //	Extract input parameters
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        // parameters
        const fp a2 = 2.0f * d_par_gpu.alpha * d_par_gpu.alpha;

        // home box
        const int first_i = d_box_gpu[bx].offset;
        FOUR_VECTOR * __restrict__ rA = &d_rv_gpu[first_i];
        FOUR_VECTOR * __restrict__ fA = &d_fv_gpu[first_i];
        __shared__ FOUR_VECTOR rA_shared[100];

        // nei box
        int pointer;
        int k = 0;
        int first_j;
        FOUR_VECTOR * __restrict__ rB;
        fp * __restrict__ qB;
        int j = 0;
        __shared__ FOUR_VECTOR rB_shared[100];
        __shared__ fp qB_shared[100];

        // common
        fp r2;
        fp u2;
        fp vij;
        fp fs;
        fp fxij;
        fp fyij;
        fp fzij;
        THREE_VECTOR d;

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //	Home box
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        //----------------------------------------------------------------------------------------------------------------------------------140
        //	Copy to shared memory with vectorized loads
        //----------------------------------------------------------------------------------------------------------------------------------140

        // Use vectorized loads for better memory throughput
        float4* rA_vec = (float4*)rA;
        float4* rA_shared_vec = (float4*)rA_shared;
        
        for (int i = tx; i < 25; i += NUMBER_THREADS) { // 100/4 = 25
            rA_shared_vec[i] = rA_vec[i];
        }
        
        __syncthreads();

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //	nei box loop
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        // loop over neiing boxes of home box
        for (k = 0; k < (1 + d_box_gpu[bx].nn); k++) {

            //----------------------------------------50
            //	nei box - get pointer to the right box
            //----------------------------------------50

            pointer = (k == 0) ? bx : d_box_gpu[bx].nei[k - 1].number;

            //----------------------------------------------------------------------------------------------------------------------------------140
            //	Setup parameters
            //----------------------------------------------------------------------------------------------------------------------------------140

            // nei box - box parameters
            first_j = d_box_gpu[pointer].offset;

            // nei box - distance, (force), charge and (type) parameters
            rB = &d_rv_gpu[first_j];
            qB = &d_qv_gpu[first_j];

            //----------------------------------------------------------------------------------------------------------------------------------140
            //	Copy to shared memory with vectorized loads
            //----------------------------------------------------------------------------------------------------------------------------------140

            // Vectorized loads for rB
            float4* rB_vec = (float4*)rB;
            float4* rB_shared_vec = (float4*)rB_shared;
            
            for (int i = tx; i < 25; i += NUMBER_THREADS) {
                rB_shared_vec[i] = rB_vec[i];
            }
            
            // Load qB values
            for (int i = tx; i < 100; i += NUMBER_THREADS) {
                qB_shared[i] = qB[i];
            }
            
            __syncthreads();

            //----------------------------------------------------------------------------------------------------------------------------------140
            //	Calculation with warp-level optimizations
            //----------------------------------------------------------------------------------------------------------------------------------140

            // Process particles using all threads in block
            for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
                const FOUR_VECTOR rA_local = rA_shared[i];
                
                // Use register blocking to reduce shared memory accesses
                #pragma unroll 4
                for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
                    const FOUR_VECTOR rB_local = rB_shared[j];
                    const fp qB_local = qB_shared[j];

                    r2 = rA_local.v + rB_local.v - 
                         (rA_local.x * rB_local.x + rA_local.y * rB_local.y + rA_local.z * rB_local.z);
                    u2 = a2 * r2;
                    vij = expf(-u2); // Use fast math function
                    fs = 2.0f * vij;

                    d.x = rA_local.x - rB_local.x;
                    d.y = rA_local.y - rB_local.y;
                    d.z = rA_local.z - rB_local.z;

                    fxij = fs * d.x;
                    fyij = fs * d.y;
                    fzij = fs * d.z;

                    // Use atomic adds to handle race conditions properly
                    atomicAdd(&fA[i].v, qB_local * vij);
                    atomicAdd(&fA[i].x, qB_local * fxij);
                    atomicAdd(&fA[i].y, qB_local * fyij);
                    atomicAdd(&fA[i].z, qB_local * fzij);
                }
            }

            __syncthreads();
        }
    }
}
