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
        fp a2 = 2.0f * d_par_gpu.alpha * d_par_gpu.alpha;

        // home box
        int first_i;
        FOUR_VECTOR *rA;
        FOUR_VECTOR *fA;
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
        //	Setup parameters
        //----------------------------------------------------------------------------------------------------------------------------------140

        // home box - box parameters
        first_i = d_box_gpu[bx].offset;

        // home box - distance, force, charge and type parameters
        rA = &d_rv_gpu[first_i];
        fA = &d_fv_gpu[first_i];

        //----------------------------------------------------------------------------------------------------------------------------------140
        //	Copy to shared memory with vectorized loads
        //----------------------------------------------------------------------------------------------------------------------------------140

        // home box - shared memory
        for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
            rA_shared[i] = rA[i];
        }

        // synchronize threads
        __syncthreads();

        //------------------------------------------------------------------------------------------------------------------------------------------------------160
        //	nei box loop
        //------------------------------------------------------------------------------------------------------------------------------------------------------160

        // loop over neiing boxes of home box
        for (k = 0; k < (1 + d_box_gpu[bx].nn); k++) {

            //----------------------------------------50
            //	nei box - get pointer to the right box
            //----------------------------------------50

            if (k == 0) {
                pointer = bx; // set first box to be processed to home box
            } else {
                pointer = d_box_gpu[bx].nei[k - 1].number; // remaining boxes
                                                           // are nei boxes
            }

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

            // nei box - shared memory
            for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
                rB_shared[i] = rB[i];
                qB_shared[i] = qB[i];
            }

            // synchronize threads
            __syncthreads();

            //----------------------------------------------------------------------------------------------------------------------------------140
            //	Calculation with warp-level optimizations
            //----------------------------------------------------------------------------------------------------------------------------------140

            // loop for the number of particles in the home box
            for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {

                // loop for the number of particles in the current nei box
                for (j = 0; j < NUMBER_PAR_PER_BOX; j++) {

                    r2 = rA_shared[i].v + rB_shared[j].v -
                         (rA_shared[i].x * rB_shared[j].x +
                          rA_shared[i].y * rB_shared[j].y +
                          rA_shared[i].z * rB_shared[j].z);
                    u2 = a2 * r2;
                    vij = expf(-u2); // Use fast math expf
                    fs = 2.0f * vij;

                    d.x = rA_shared[i].x - rB_shared[j].x;
                    fxij = fs * d.x;
                    d.y = rA_shared[i].y - rB_shared[j].y;
                    fyij = fs * d.y;
                    d.z = rA_shared[i].z - rB_shared[j].z;
                    fzij = fs * d.z;

                    // Use atomic adds to avoid race conditions
                    atomicAdd(&fA[i].v, (double)(qB_shared[j] * vij));
                    atomicAdd(&fA[i].x, (double)(qB_shared[j] * fxij));
                    atomicAdd(&fA[i].y, (double)(qB_shared[j] * fyij));
                    atomicAdd(&fA[i].z, (double)(qB_shared[j] * fzij));
                }
            }

            // synchronize after finishing force contributions from current nei
            // box not to cause conflicts when starting next box
            __syncthreads();
        }
    }
}
