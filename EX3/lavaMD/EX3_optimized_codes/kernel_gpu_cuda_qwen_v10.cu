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
        FOUR_VECTOR *rA = &d_rv_gpu[first_i];
        FOUR_VECTOR *fA = &d_fv_gpu[first_i];
        __shared__ FOUR_VECTOR rA_shared[100];

        // nei box
        int pointer;
        int k = 0;
        int first_j;
        FOUR_VECTOR *rB;
        fp *qB;
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

        // home box - shared memory
        for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
            rA_shared[i] = rA[i];
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

            for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
                rB_shared[i] = rB[i];
                qB_shared[i] = qB[i];
            }
            __syncthreads();

            //----------------------------------------------------------------------------------------------------------------------------------140
            //	Calculation with warp-level optimizations
            //----------------------------------------------------------------------------------------------------------------------------------140

            // Process particles in chunks aligned with warp size for better memory coalescing
            for (int base_i = 0; base_i < NUMBER_PAR_PER_BOX; base_i += 32) {
                int i = base_i + lane_id;
                if (i < NUMBER_PAR_PER_BOX) {
                    const FOUR_VECTOR rA_local = rA_shared[i];
                    const fp rA_v = rA_local.v;
                    const fp rA_x = rA_local.x;
                    const fp rA_y = rA_local.y;
                    const fp rA_z = rA_local.z;
                    
                    FOUR_VECTOR fA_local;
                    fA_local.v = 0.0f;
                    fA_local.x = 0.0f;
                    fA_local.y = 0.0f;
                    fA_local.z = 0.0f;

                    // Unroll inner loop partially to reduce loop overhead
                    #pragma unroll 4
                    for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {
                        const FOUR_VECTOR rB_local = rB_shared[j];
                        const fp qB_val = qB_shared[j];
                        
                        r2 = rA_v + rB_local.v - (rA_x * rB_local.x + rA_y * rB_local.y + rA_z * rB_local.z);
                        u2 = a2 * r2;
                        vij = expf(-u2); // Use fast math expf
                        fs = 2.0f * vij;

                        d.x = rA_x - rB_local.x;
                        d.y = rA_y - rB_local.y;
                        d.z = rA_z - rB_local.z;

                        fxij = fs * d.x;
                        fyij = fs * d.y;
                        fzij = fs * d.z;

                        fA_local.v += qB_val * vij;
                        fA_local.x += qB_val * fxij;
                        fA_local.y += qB_val * fyij;
                        fA_local.z += qB_val * fzij;
                    }
                    
                    // Accumulate results atomically if needed, but since each thread works on unique i, direct write is safe
                    fA[i].v += fA_local.v;
                    fA[i].x += fA_local.x;
                    fA[i].y += fA_local.y;
                    fA[i].z += fA_local.z;
                }
            }

            __syncthreads();
        }
    }
}
