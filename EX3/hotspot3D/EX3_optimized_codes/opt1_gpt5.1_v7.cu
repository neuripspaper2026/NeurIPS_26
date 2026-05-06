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

    int c = i + j * nx;
    int xy = nx * ny;

    int W = (i == 0)      ? c : c - 1;
    int E = (i == nx - 1) ? c : c + 1;
    int N = (j == 0)      ? c : c - nx;
    int S = (j == ny - 1) ? c : c + nx;

    // Use temporaries to keep frequently used data in registers
    float t_center = tIn[c];
    float t_w = tIn[W];
    float t_e = tIn[E];
    float t_n = tIn[N];
    float t_s = tIn[S];
    float t_below = t_center;
    float t_above = tIn[c + xy];
    float power = p[c];

    // Precompute constant term
    float amb_term = ct * amb_temp;

    // First layer (k = 0)
    tOut[c] = cc * t_center +
              cw * t_w +
              ce * t_e +
              cs * t_s +
              cn * t_n +
              cb * t_below +
              ct * t_above +
              sdc * power +
              amb_term;

    c      += xy;
    W      += xy;
    E      += xy;
    N      += xy;
    S      += xy;

    // Main z-loop: 1 .. nz-2
    for (int k = 1; k < nz - 1; ++k) {
        t_below  = t_center;
        t_center = t_above;
        t_above  = tIn[c + xy];

        t_w = tIn[W];
        t_e = tIn[E];
        t_n = tIn[N];
        t_s = tIn[S];
        power = p[c];

        tOut[c] = cc * t_center +
                  cw * t_w +
                  ce * t_e +
                  cs * t_s +
                  cn * t_n +
                  cb * t_below +
                  ct * t_above +
                  sdc * power +
                  amb_term;

        c      += xy;
        W      += xy;
        E      += xy;
        N      += xy;
        S      += xy;
    }

    // Last layer (k = nz-1)
    t_below  = t_center;
    t_center = t_above;

    t_w = tIn[W];
    t_e = tIn[E];
    t_n = tIn[N];
    t_s = tIn[S];
    power = p[c];

    tOut[c] = cc * t_center +
              cw * t_w +
              ce * t_e +
              cs * t_s +
              cn * t_n +
              cb * t_below +
              ct * t_center +
              sdc * power +
              amb_term;
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
