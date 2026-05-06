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

    // Shared memory for home box and neighbor box
    __shared__ FOUR_VECTOR rA_shared[NUMBER_PAR_PER_BOX];
    __shared__ FOUR_VECTOR rB_shared[NUMBER_PAR_PER_BOX];
    __shared__ fp qB_shared[NUMBER_PAR_PER_BOX];

    // home box parameters
    int first_i = d_box_gpu[bx].offset;
    FOUR_VECTOR *rA = &d_rv_gpu[first_i];
    FOUR_VECTOR *fA = &d_fv_gpu[first_i];

    // Load home box data to shared memory with coalesced access
    for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
        rA_shared[i] = rA[i];
    }
    __syncthreads();

    // Register accumulation for forces
    fp fA_v_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    fp fA_x_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    fp fA_y_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    fp fA_z_local[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Process each particle assigned to this thread (4 particles per thread)
    int num_particles_per_thread = (NUMBER_PAR_PER_BOX + NUMBER_THREADS - 1) / NUMBER_THREADS;
    if (num_particles_per_thread > 4) num_particles_per_thread = 4;

    // Loop over neighboring boxes
    for (int k = 0; k < (1 + d_box_gpu[bx].nn); k++) {

        int pointer = (k == 0) ? bx : d_box_gpu[bx].nei[k - 1].number;
        int first_j = d_box_gpu[pointer].offset;

        FOUR_VECTOR *rB = &d_rv_gpu[first_j];
        fp *qB = &d_qv_gpu[first_j];

        // Load neighbor box data with coalesced access
        for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
            rB_shared[i] = rB[i];
            qB_shared[i] = qB[i];
        }
        __syncthreads();

        // Each thread processes multiple particles (up to 4)
        for (int p = 0; p < num_particles_per_thread; p++) {
            int wtx = tx + p * NUMBER_THREADS;
            if (wtx >= NUMBER_PAR_PER_BOX) break;

            FOUR_VECTOR rA_local = rA_shared[wtx];

            // Inner loop over all particles in neighbor box
            #pragma unroll 4
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

                fp fxij = fs * dx;
                fp fyij = fs * dy;
                fp fzij = fs * dz;

                fA_v_local[p] += qB_local * vij;
                fA_x_local[p] += qB_local * fxij;
                fA_y_local[p] += qB_local * fyij;
                fA_z_local[p] += qB_local * fzij;
            }
        }
        __syncthreads();
    }

    // Write back accumulated forces with coalesced access
    for (int p = 0; p < num_particles_per_thread; p++) {
        int wtx = tx + p * NUMBER_THREADS;
        if (wtx < NUMBER_PAR_PER_BOX) {
            fA[wtx].v += (double)fA_v_local[p];
            fA[wtx].x += (double)fA_x_local[p];
            fA[wtx].y += (double)fA_y_local[p];
            fA[wtx].z += (double)fA_z_local[p];
        }
    }
}
