__global__ void kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu,
                                box_str *d_box_gpu, FOUR_VECTOR *d_rv_gpu,
                                fp *d_qv_gpu, FOUR_VECTOR *d_fv_gpu) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    if (bx >= d_dim_gpu.number_boxes) {
        return;
    }

    // parameters
    fp a2 = 2.0f * d_par_gpu.alpha * d_par_gpu.alpha;

    // home box
    int first_i = d_box_gpu[bx].offset;
    FOUR_VECTOR *rA = &d_rv_gpu[first_i];
    FOUR_VECTOR *fA = &d_fv_gpu[first_i];
    
    __shared__ FOUR_VECTOR rA_shared[100];
    __shared__ FOUR_VECTOR rB_shared[100];
    __shared__ fp qB_shared[100];

    // Load home box data to shared memory with coalesced access
    #pragma unroll 4
    for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
        rA_shared[i] = rA[i];
    }
    __syncthreads();

    // Accumulate forces in registers
    fp fv_acc = 0.0f;
    fp fx_acc = 0.0f;
    fp fy_acc = 0.0f;
    fp fz_acc = 0.0f;

    // Process each particle assigned to this thread
    for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
        FOUR_VECTOR rA_local = rA_shared[i];
        
        // Loop over neighboring boxes
        for (int k = 0; k < (1 + d_box_gpu[bx].nn); k++) {
            int pointer = (k == 0) ? bx : d_box_gpu[bx].nei[k - 1].number;
            int first_j = d_box_gpu[pointer].offset;
            FOUR_VECTOR *rB = &d_rv_gpu[first_j];
            fp *qB = &d_qv_gpu[first_j];

            // Cooperative loading of neighbor box data
            for (int idx = threadIdx.x; idx < NUMBER_PAR_PER_BOX; idx += NUMBER_THREADS) {
                rB_shared[idx] = rB[idx];
                qB_shared[idx] = qB[idx];
            }
            __syncthreads();

            // Compute interactions with all particles in neighbor box
            #pragma unroll 8
            for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
                FOUR_VECTOR rB_local = rB_shared[j];
                fp qB_local = qB_shared[j];

                // Calculate distance and force
                fp dx = rA_local.x - rB_local.x;
                fp dy = rA_local.y - rB_local.y;
                fp dz = rA_local.z - rB_local.z;
                
                fp r2 = rA_local.v + rB_local.v - (dx * dx + dy * dy + dz * dz);
                fp u2 = a2 * r2;
                fp vij = expf(-u2);
                fp fs = 2.0f * vij;

                fp qB_vij = qB_local * vij;
                fp qB_fs = qB_local * fs;

                fv_acc += qB_vij;
                fx_acc += qB_fs * dx;
                fy_acc += qB_fs * dy;
                fz_acc += qB_fs * dz;
            }
            __syncthreads();
        }

        // Write accumulated forces back to global memory
        atomicAdd(&fA[i].v, (double)fv_acc);
        atomicAdd(&fA[i].x, (double)fx_acc);
        atomicAdd(&fA[i].y, (double)fy_acc);
        atomicAdd(&fA[i].z, (double)fz_acc);

        // Reset accumulators for next particle
        fv_acc = 0.0f;
        fx_acc = 0.0f;
        fy_acc = 0.0f;
        fz_acc = 0.0f;
    }
}
