long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

__global__ void hotspotOpt1(float *p, float *tIn, float *tOut, float sdc,
                            int nx, int ny, int nz, float ce, float cw,
                            float cn, float cs, float ct, float cb, float cc) {
    const float amb_temp = 80.0f;

    // Compute global indices
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    // Guard against out-of-range threads (for non-perfect grid sizes)
    if (i >= nx || j >= ny) {
        return;
    }

    int c = i + j * nx;
    int xy = nx * ny;

    // Preload center value
    float center = tIn[c];

    // Compute neighbor indices with boundary clamping (avoid branches later)
    int iW = max(i - 1, 0);
    int iE = min(i + 1, nx - 1);
    int jN = max(j - 1, 0);
    int jS = min(j + 1, ny - 1);

    int W = iW + j *       nx;
    int E = iE + j *       nx;
    int N = i  + jN *      nx;
    int S = i  + jS *      nx;

    // z-1, z, z+1 temps
    float temp1 = center;
    float temp2 = center;
    float temp3 = tIn[c + xy];

    // Common term for ambient temperature
    float ct_amb = ct * amb_temp;

    // First slice (k = 0)
    tOut[c] = cc * temp2 +
              cw * tIn[W] + ce * tIn[E] +
              cs * tIn[S] + cn * tIn[N] +
              cb * temp1 + ct * temp3 +
              sdc * p[c] + ct_amb;

    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;

    // Main z loop
    #pragma unroll 4
    for (int k = 1; k < nz - 1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c + xy];

        float valW = tIn[W];
        float valE = tIn[E];
        float valS = tIn[S];
        float valN = tIn[N];
        float valP = p[c];

        tOut[c] = cc * temp2 +
                  cw * valW + ce * valE +
                  cs * valS + cn * valN +
                  cb * temp1 + ct * temp3 +
                  sdc * valP + ct_amb;

        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }

    // Last slice (k = nz - 1)
    temp1 = temp2;
    temp2 = temp3;

    float valW_last = tIn[W];
    float valE_last = tIn[E];
    float valS_last = tIn[S];
    float valN_last = tIn[N];
    float valP_last = p[c];

    tOut[c] = cc * temp2 +
              cw * valW_last + ce * valE_last +
              cs * valS_last + cn * valN_last +
              cb * temp1 + ct * temp3 +
              sdc * valP_last + ct_amb;
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
