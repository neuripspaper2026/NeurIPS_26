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
    __shared__ FOUR_VECTOR rA_shared[100];
    __shared__ FOUR_VECTOR rB_shared[100];
    __shared__ fp qB_shared[100];

    // Home box setup
    int first_i = d_box_gpu[bx].offset;
    FOUR_VECTOR *rA = &d_rv_gpu[first_i];
    FOUR_VECTOR *fA = &d_fv_gpu[first_i];

    // Load home box data into shared memory with coalesced access
    for (int i = tx; i < NUMBER_PAR_PER_BOX; i += NUMBER_THREADS) {
        rA_shared[i] = rA[i];
    }
    __syncthreads();

    // Register arrays for accumulating forces per thread
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

        // Each thread processes 4 particles from home box for better instruction-level parallelism
        for (int i_base = tx * 4; i_base < NUMBER_PAR_PER_BOX; i_base += NUMBER_THREADS * 4) {
            #pragma unroll
            for (int i_offset = 0; i_offset < 4; i_offset++) {
                int i = i_base + i_offset;
                if (i >= NUMBER_PAR_PER_BOX) break;

                fp rA_v = rA_shared[i].v;
                fp rA_x = rA_shared[i].x;
                fp rA_y = rA_shared[i].y;
                fp rA_z = rA_shared[i].z;

                fp local_fv = 0.0f;
                fp local_fx = 0.0f;
                fp local_fy = 0.0f;
                fp local_fz = 0.0f;

                // Inner loop over neighbor particles with unrolling
                #pragma unroll 4
                for (int j = 0; j < NUMBER_PAR_PER_BOX; j++) {
                    fp rB_v = rB_shared[j].v;
                    fp rB_x = rB_shared[j].x;
                    fp rB_y = rB_shared[j].y;
                    fp rB_z = rB_shared[j].z;
                    fp qB_val = qB_shared[j];

                    fp dx = rA_x - rB_x;
                    fp dy = rA_y - rB_y;
                    fp dz = rA_z - rB_z;

                    fp dot_product = rA_x * rB_x + rA_y * rB_y + rA_z * rB_z;
                    fp r2 = rA_v + rB_v - dot_product;
                    fp u2 = a2 * r2;
                    fp vij = expf(-u2);
                    fp fs = 2.0f * vij;

                    local_fv += qB_val * vij;
                    local_fx += qB_val * fs * dx;
                    local_fy += qB_val * fs * dy;
                    local_fz += qB_val * fs * dz;
                }

                fA_v[i_offset] += local_fv;
                fA_x[i_offset] += local_fx;
                fA_y[i_offset] += local_fy;
                fA_z[i_offset] += local_fz;
            }
        }
        __syncthreads();
    }

    // Write accumulated forces back to global memory with coalesced access
    for (int i_base = tx * 4; i_base < NUMBER_PAR_PER_BOX; i_base += NUMBER_THREADS * 4) {
        #pragma unroll
        for (int i_offset = 0; i_offset < 4; i_offset++) {
            int i = i_base + i_offset;
            if (i >= NUMBER_PAR_PER_BOX) break;

            atomicAdd(&fA[i].v, (double)fA_v[i_offset]);
            atomicAdd(&fA[i].x, (double)fA_x[i_offset]);
            atomicAdd(&fA[i].y, (double)fA_y[i_offset]);
            atomicAdd(&fA[i].z, (double)fA_z[i_offset]);
        }
    }
}
