long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

__global__ void hotspotOpt1(float *p, float *tIn, float *tOut, float sdc,
                            int nx, int ny, int nz, float ce, float cw,
                            float cn, float cs, float ct, float cb, float cc) {
    const float amb_temp = 80.0f;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    const int xy = nx * ny;
    int c = i + j * nx;

    int W = (i == 0)      ? c : c - 1;
    int E = (i == nx - 1) ? c : c + 1;
    int N = (j == 0)      ? c : c - nx;
    int S = (j == ny - 1) ? c : c + nx;

    // Preload first three z-slices for this (i,j) column
    float temp1 = tIn[c];
    float temp2 = temp1;
    float temp3 = (nz > 1) ? tIn[c + xy] : temp2;

    // Compute first layer (k = 0)
    float tW = tIn[W];
    float tE = tIn[E];
    float tN = tIn[N];
    float tS = tIn[S];
    float pval = p[c];

    tOut[c] = cc * temp2 + cw * tW + ce * tE + cs * tS +
              cn * tN + cb * temp1 + ct * temp3 + sdc * pval +
              ct * amb_temp;

    // Advance indices to next z-layer
    c  += xy;
    W  += xy;
    E  += xy;
    N  += xy;
    S  += xy;

    // Main z-loop (1 .. nz-2), unrolled by 2 for better ILP
    int k = 1;
    const int nzm1 = nz - 1;

    for (; k + 1 < nzm1; k += 2) {
        // -------- iteration k --------
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c + xy];

        tW   = tIn[W];
        tE   = tIn[E];
        tN   = tIn[N];
        tS   = tIn[S];
        pval = p[c];

        tOut[c] = cc * temp2 + cw * tW + ce * tE + cs * tS +
                  cn * tN + cb * temp1 + ct * temp3 + sdc * pval +
                  ct * amb_temp;

        int c_next  = c  + xy;
        int W_next  = W  + xy;
        int E_next  = E  + xy;
        int N_next  = N  + xy;
        int S_next  = S  + xy;

        // -------- iteration k+1 (software pipelined) --------
        float temp1_n = temp2;
        float temp2_n = temp3;
        float temp3_n = tIn[c_next + xy];

        float tW_n   = tIn[W_next];
        float tE_n   = tIn[E_next];
        float tN_n   = tIn[N_next];
        float tS_n   = tIn[S_next];
        float pval_n = p[c_next];

        tOut[c_next] = cc * temp2_n + cw * tW_n + ce * tE_n + cs * tS_n +
                       cn * tN_n + cb * temp1_n + ct * temp3_n + sdc * pval_n +
                       ct * amb_temp;

        // Commit state for next double-iteration
        c   = c_next + xy;
        W   = W_next + xy;
        E   = E_next + xy;
        N   = N_next + xy;
        S   = S_next + xy;
        temp1 = temp1_n;
        temp2 = temp2_n;
        temp3 = temp3_n;
    }

    // Handle remaining middle layer if nz-2 is odd
    for (; k < nzm1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c + xy];

        tW   = tIn[W];
        tE   = tIn[E];
        tN   = tIn[N];
        tS   = tIn[S];
        pval = p[c];

        tOut[c] = cc * temp2 + cw * tW + ce * tE + cs * tS +
                  cn * tN + cb * temp1 + ct * temp3 + sdc * pval +
                  ct * amb_temp;

        c  += xy;
        W  += xy;
        E  += xy;
        N  += xy;
        S  += xy;
    }

    // Last layer (k = nz-1)
    temp1 = temp2;
    temp2 = temp3;

    tW   = tIn[W];
    tE   = tIn[E];
    tN   = tIn[N];
    tS   = tIn[S];
    pval = p[c];

    tOut[c] = cc * temp2 + cw * tW + ce * tE + cs * tS +
              cn * tN + cb * temp1 + ct * temp3 + sdc * pval +
              ct * amb_temp;
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
