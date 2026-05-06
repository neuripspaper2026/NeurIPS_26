__global__ void kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu,
                                box_str *d_box_gpu, FOUR_VECTOR *d_rv_gpu,
                                fp *d_qv_gpu, FOUR_VECTOR *d_fv_gpu) {

    int bx = blockIdx.x;
    int tx = threadIdx.x;

    if (bx >= d_dim_gpu.number_boxes) {
        return;
    }

    // Parameters
    fp a2 = 2.0f * d_par_gpu.alpha * d_par_gpu.alpha;

    // Shared memory for home box and neighbor box
    __shared__ FOUR_VECTOR rA_shared[NUMBER_PAR_PER_BOX];
    __shared__ FOUR_VECTOR rB_shared[NUMBER_PAR_PER_BOX];
    __shared__ fp qB_shared[NUMBER_PAR_PER_BOX];

    // Home box parameters
    int first_i = d_box_gpu[bx].offset;
    FOUR_VECTOR *rA = &d_rv_gpu[first_i];
    FOUR_VECTOR *fA = &d_fv_gpu[first_i];

    // Load home box data to shared memory with coalesced access
    for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
        rA_shared[i] = rA[i];
    }
    __syncthreads();

    // Register arrays to accumulate forces for each particle this thread handles
    fp fA_v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    fp fA_x[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    fp fA_y[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    fp fA_z[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Loop over neighboring boxes
    for (int k = 0; k < (1 + d_box_gpu[bx].nn); k++) {

        int pointer = (k == 0) ? bx : d_box_gpu[bx].nei[k - 1].number;
        int first_j = d_box_gpu[pointer].offset;

        FOUR_VECTOR *rB = &d_rv_gpu[first_j];
        fp *qB = &d_qv_gpu[first_j];

        // Load neighbor box data to shared memory with coalesced access
        for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
            rB_shared[i] = rB[i];
            qB_shared[i] = qB[i];
        }
        __syncthreads();

        // Each thread processes multiple particles (unrolled by 4)
        int wtx = tx;
        while (wtx < NUMBER_PAR_PER_BOX) {
            int idx = wtx / NUMBER_THREADS;
            if (idx >= 4) break;

            FOUR_VECTOR rA_local = rA_shared[wtx];

            // Inner loop over neighbor particles
            #pragma unroll 8
            for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
                FOUR_VECTOR rB_local = rB_shared[j];
                fp qB_local = qB_shared[j];

                fp dx = rA_local.x - rB_local.x;
                fp dy = rA_local.y - rB_local.y;
                fp dz = rA_local.z - rB_local.z;

                fp r2 = rA_local.v + rB_local.v - (dx * dx + dy * dy + dz * dz);
                fp u2 = a2 * r2;
                fp vij = expf(-u2);
                fp fs = 2.0f * vij;

                fp qB_vij = qB_local * vij;
                fp qB_fs = qB_local * fs;

                fA_v[idx] += qB_vij;
                fA_x[idx] += qB_fs * dx;
                fA_y[idx] += qB_fs * dy;
                fA_z[idx] += qB_fs * dz;
            }

            wtx += NUMBER_THREADS;
        }

        __syncthreads();
    }

    // Write accumulated forces back to global memory
    int wtx = tx;
    int idx = 0;
    while (wtx < NUMBER_PAR_PER_BOX && idx < 4) {
        atomicAdd(&fA[wtx].v, (double)fA_v[idx]);
        atomicAdd(&fA[wtx].x, (double)fA_x[idx]);
        atomicAdd(&fA[wtx].y, (double)fA_y[idx]);
        atomicAdd(&fA[wtx].z, (double)fA_z[idx]);
        wtx += NUMBER_THREADS;
        idx++;
    }
}
