long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

__global__ void hotspotOpt1(float *p, float *tIn, float *tOut, float sdc,
                            int nx, int ny, int nz, float ce, float cw,
                            float cn, float cs, float ct, float cb, float cc) {
    float amb_temp = 80.0;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int c = i + j * nx;
    int xy = nx * ny;

    // Shared memory for halo regions
    extern __shared__ float shared_t[];

    // Load data into shared memory with halo
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Calculate shared memory dimensions
    int shared_width = blockDim.x + 2;
    int shared_height = blockDim.y + 2;

    // Load center data
    if (i < nx && j < ny) {
        for (int k = 0; k < nz; k++) {
            shared_t[(ty + 1) * shared_width + (tx + 1) + k * shared_width * shared_height] = tIn[c + k * xy];
        }
    }

    // Load halo data
    // Left halo
    if (tx == 0 && i > 0) {
        int halo_idx = (i - 1) + j * nx;
        for (int k = 0; k < nz; k++) {
            shared_t[ty * shared_width + tx + k * shared_width * shared_height] = tIn[halo_idx + k * xy];
        }
    }
    // Right halo
    if (tx == blockDim.x - 1 && i < nx - 1) {
        int halo_idx = (i + 1) + j * nx;
        for (int k = 0; k < nz; k++) {
            shared_t[(ty + 2) * shared_width + (tx + 2) + k * shared_width * shared_height] = tIn[halo_idx + k * xy];
        }
    }
    // Top halo
    if (ty == 0 && j > 0) {
        int halo_idx = i + (j - 1) * nx;
        for (int k = 0; k < nz; k++) {
            shared_t[ty * shared_width + (tx + 1) + k * shared_width * shared_height] = tIn[halo_idx + k * xy];
        }
    }
    // Bottom halo
    if (ty == blockDim.y - 1 && j < ny - 1) {
        int halo_idx = i + (j + 1) * nx;
        for (int k = 0; k < nz; k++) {
            shared_t[(ty + 2) * shared_width + (tx + 1) + k * shared_width * shared_height] = tIn[halo_idx + k * xy];
        }
    }

    // Corners
    if (tx == 0 && ty == 0 && i > 0 && j > 0) {
        int halo_idx = (i - 1) + (j - 1) * nx;
        for (int k = 0; k < nz; k++) {
            shared_t[ty * shared_width + tx + k * shared_width * shared_height] = tIn[halo_idx + k * xy];
        }
    }
    if (tx == blockDim.x - 1 && ty == 0 && i < nx - 1 && j > 0) {
        int halo_idx = (i + 1) + (j - 1) * nx;
        for (int k = 0; k < nz; k++) {
            shared_t[ty * shared_width + (tx + 2) + k * shared_width * shared_height] = tIn[halo_idx + k * xy];
        }
    }
    if (tx == 0 && ty == blockDim.y - 1 && i > 0 && j < ny - 1) {
        int halo_idx = (i - 1) + (j + 1) * nx;
        for (int k = 0; k < nz; k++) {
            shared_t[(ty + 2) * shared_width + tx + k * shared_width * shared_height] = tIn[halo_idx + k * xy];
        }
    }
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1 && i < nx - 1 && j < ny - 1) {
        int halo_idx = (i + 1) + (j + 1) * nx;
        for (int k = 0; k < nz; k++) {
            shared_t[(ty + 2) * shared_width + (tx + 2) + k * shared_width * shared_height] = tIn[halo_idx + k * xy];
        }
    }

    __syncthreads();

    if (i >= nx || j >= ny) return;

    int W = (i == 0) ? 0 : -1;
    int E = (i == nx - 1) ? 0 : 1;
    int N = (j == 0) ? 0 : -shared_width;
    int S = (j == ny - 1) ? 0 : shared_width;

    float temp1, temp2, temp3;
    temp1 = temp2 = shared_t[(ty + 1) * shared_width + (tx + 1)];
    temp3 = shared_t[(ty + 1) * shared_width + (tx + 1) + shared_width * shared_height];
    
    tOut[c] = cc * temp2 + cw * shared_t[(ty + 1) * shared_width + (tx + 1) + W] + 
              ce * shared_t[(ty + 1) * shared_width + (tx + 1) + E] + 
              cs * shared_t[(ty + 1) * shared_width + (tx + 1) + S] +
              cn * shared_t[(ty + 1) * shared_width + (tx + 1) + N] + 
              cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
    
    c += xy;

    for (int k = 1; k < nz - 1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = shared_t[(ty + 1) * shared_width + (tx + 1) + (k + 1) * shared_width * shared_height];
        
        tOut[c] = cc * temp2 + cw * shared_t[(ty + 1) * shared_width + (tx + 1) + W + k * shared_width * shared_height] + 
                  ce * shared_t[(ty + 1) * shared_width + (tx + 1) + E + k * shared_width * shared_height] + 
                  cs * shared_t[(ty + 1) * shared_width + (tx + 1) + S + k * shared_width * shared_height] +
                  cn * shared_t[(ty + 1) * shared_width + (tx + 1) + N + k * shared_width * shared_height] + 
                  cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
        c += xy;
    }
    
    temp1 = temp2;
    temp2 = temp3;
    tOut[c] = cc * temp2 + cw * shared_t[(ty + 1) * shared_width + (tx + 1) + W + (nz-1) * shared_width * shared_height] + 
              ce * shared_t[(ty + 1) * shared_width + (tx + 1) + E + (nz-1) * shared_width * shared_height] + 
              cs * shared_t[(ty + 1) * shared_width + (tx + 1) + S + (nz-1) * shared_width * shared_height] +
              cn * shared_t[(ty + 1) * shared_width + (tx + 1) + N + (nz-1) * shared_width * shared_height] + 
              cb * temp1 + ct * temp3 + sdc * p[c] + ct * amb_temp;
    return;
}

void hotspot_opt1(float *p, float *tIn, float *tOut, int nx, int ny, int nz,
                  float Cap, float Rx, float Ry, float Rz, float dt,
                  int numiter) {
    float ce, cw, cn, cs, ct, cb, cc;
    float stepDivCap = dt / Cap;
    ce = cw = stepDivCap / Rx;
    cn = cs = stepDivCap / Ry;
    ct = cb = stepDivCap / Rz;

    cc = 1.0 - (2.0 * ce + 2.0 * cn + 3.0 * ct);

    size_t s = sizeof(float) * nx * ny * nz;
    float *tIn_d, *tOut_d, *p_d;
    cudaMalloc((void **)&p_d, s);
    cudaMalloc((void **)&tIn_d, s);
    cudaMalloc((void **)&tOut_d, s);
    cudaMemcpy(tIn_d, tIn, s, cudaMemcpyHostToDevice);
    cudaMemcpy(p_d, p, s, cudaMemcpyHostToDevice);

    cudaFuncSetCacheConfig(hotspotOpt1, cudaFuncCachePreferL1);

    dim3 block_dim(64, 4, 1);
    dim3 grid_dim(nx / 64, ny / 4, 1);

    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);
    
    for (int i = 0; i < numiter; ++i) {
        hotspotOpt1<<<grid_dim, block_dim>>>(p_d, tIn_d, tOut_d, stepDivCap, nx,
                                             ny, nz, ce, cw, cn, cs, ct, cb,
                                             cc);
        float *t = tIn_d;
        tIn_d = tOut_d;
        tOut_d = t;
    }
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    
    extern double g_kernel_time;
    g_kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) + 
                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
    printf("Time: %.3f (s)\n", g_kernel_time);
    cudaMemcpy(tOut, tOut_d, s, cudaMemcpyDeviceToHost);
    cudaFree(p_d);
    cudaFree(tIn_d);
    cudaFree(tOut_d);
    return;
}
