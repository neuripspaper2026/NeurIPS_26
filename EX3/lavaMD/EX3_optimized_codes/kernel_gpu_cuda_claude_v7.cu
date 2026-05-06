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

    // Load home box data into shared memory with coalesced access
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
    int num_neighbors = 1 + d_box_gpu[bx].nn;
    for (int k = 0; k < num_neighbors; k++) {

        int pointer = (k == 0) ? bx : d_box_gpu[bx].nei[k - 1].number;
        int first_j = d_box_gpu[pointer].offset;

        FOUR_VECTOR *rB = &d_rv_gpu[first_j];
        fp *qB = &d_qv_gpu[first_j];

        // Load neighbor box data into shared memory with coalesced access
        for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
            rB_shared[i] = rB[i];
            qB_shared[i] = qB[i];
        }
        __syncthreads();

        // Each thread processes multiple particles (unrolled by 4)
        for (int i_base = tx; i_base < NUMBER_PAR_PER_BOX; i_base += NUMBER_THREADS * 4) {
            
            // Unroll loop: process 4 particles per thread
            #pragma unroll
            for (int unroll = 0; unroll < 4; unroll++) {
                int i = i_base + unroll * NUMBER_THREADS;
                if (i >= NUMBER_PAR_PER_BOX) break;

                FOUR_VECTOR rA_local = rA_shared[i];
                fp local_fv = 0.0f;
                fp local_fx = 0.0f;
                fp local_fy = 0.0f;
                fp local_fz = 0.0f;

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

                    local_fv += qB_local * vij;
                    local_fx += qB_local * fs * dx;
                    local_fy += qB_local * fs * dy;
                    local_fz += qB_local * fs * dz;
                }

                fA_v[unroll] += local_fv;
                fA_x[unroll] += local_fx;
                fA_y[unroll] += local_fy;
                fA_z[unroll] += local_fz;
            }
        }
        __syncthreads();
    }

    // Write back accumulated forces to global memory with coalesced access
    for (int unroll = 0; unroll < 4; unroll++) {
        int i = tx + unroll * NUMBER_THREADS;
        if (i < NUMBER_PAR_PER_BOX) {
            atomicAdd(&fA[i].v, (double)fA_v[unroll]);
            atomicAdd(&fA[i].x, (double)fA_x[unroll]);
            atomicAdd(&fA[i].y, (double)fA_y[unroll]);
            atomicAdd(&fA[i].z, (double)fA_z[unroll]);
        }
    }
}
