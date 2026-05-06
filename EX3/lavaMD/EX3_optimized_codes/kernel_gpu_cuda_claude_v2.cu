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

    __shared__ FOUR_VECTOR rA_shared[NUMBER_PAR_PER_BOX];
    __shared__ FOUR_VECTOR rB_shared[NUMBER_PAR_PER_BOX];
    __shared__ fp qB_shared[NUMBER_PAR_PER_BOX];

    // Load home box data to shared memory with coalesced access
    for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
        rA_shared[i] = rA[i];
    }
    __syncthreads();

    // Accumulate forces locally per thread
    fp fv_accum[NUMBER_PAR_PER_BOX];
    fp fx_accum[NUMBER_PAR_PER_BOX];
    fp fy_accum[NUMBER_PAR_PER_BOX];
    fp fz_accum[NUMBER_PAR_PER_BOX];

    #pragma unroll 4
    for (int i = 0; i < NUMBER_PAR_PER_BOX; i++) {
        fv_accum[i] = 0.0f;
        fx_accum[i] = 0.0f;
        fy_accum[i] = 0.0f;
        fz_accum[i] = 0.0f;
    }

    // Loop over neighboring boxes
    int num_neighbors = 1 + d_box_gpu[bx].nn;
    for (int k = 0; k < num_neighbors; k++) {

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

        // Each thread processes multiple particles from home box
        for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
            FOUR_VECTOR rA_local = rA_shared[i];
            fp fv_local = 0.0f;
            fp fx_local = 0.0f;
            fp fy_local = 0.0f;
            fp fz_local = 0.0f;

            // Inner loop over neighbor particles
            #pragma unroll 8
            for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
                FOUR_VECTOR rB_local = rB_shared[j];
                fp qB_local = qB_shared[j];

                fp r2 = rA_local.v + rB_local.v - 
                        (rA_local.x * rB_local.x + rA_local.y * rB_local.y + rA_local.z * rB_local.z);
                fp u2 = a2 * r2;
                fp vij = expf(-u2);
                fp fs = 2.0f * vij;

                fp dx = rA_local.x - rB_local.x;
                fp dy = rA_local.y - rB_local.y;
                fp dz = rA_local.z - rB_local.z;

                fv_local += qB_local * vij;
                fx_local += qB_local * fs * dx;
                fy_local += qB_local * fs * dy;
                fz_local += qB_local * fs * dz;
            }

            fv_accum[i] += fv_local;
            fx_accum[i] += fx_local;
            fy_accum[i] += fy_local;
            fz_accum[i] += fz_local;
        }
        __syncthreads();
    }

    // Write accumulated forces back to global memory with coalesced access
    for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
        fA[i].v += (double)fv_accum[i];
        fA[i].x += (double)fx_accum[i];
        fA[i].y += (double)fy_accum[i];
        fA[i].z += (double)fz_accum[i];
    }
}
