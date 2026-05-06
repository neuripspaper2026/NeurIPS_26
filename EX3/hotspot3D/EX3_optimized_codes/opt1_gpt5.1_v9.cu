long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

__global__ void hotspotOpt1(float *p, float *tIn, float *tOut, float sdc,
                            int nx, int ny, int nz, float ce, float cw,
                            float cn, float cs, float ct, float cb, float cc) {
    // Use constant instead of per-thread local
    const float amb_temp = 80.0f;

    // Cache frequently used values into registers
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bdx = blockDim.x;
    const int bdy = blockDim.y;

    int i = bdx * bx + tx;
    int j = bdy * by + ty;

    // Guard threads outside domain
    if (i >= nx || j >= ny)
        return;

    const int xy = nx * ny;
    int c = i + j * nx;

    // Precompute boundary flags to avoid repeated comparisons in z-loop
    const bool isWest  = (i == 0);
    const bool isEast  = (i == nx - 1);
    const bool isNorth = (j == 0);
    const bool isSouth = (j == ny - 1);

    int W = isWest  ? c : c - 1;
    int E = isEast  ? c : c + 1;
    int N = isNorth ? c : c - nx;
    int S = isSouth ? c : c + nx;

    // Prefetch values for rolling in z-direction
    float temp1, temp2, temp3;
    temp1 = temp2 = tIn[c];        // current layer
    temp3 = tIn[c + xy];           // next layer (+z)

    // First layer (k = 0)
    float center = temp2;
    float west   = tIn[W];
    float east   = tIn[E];
    float south  = tIn[S];
    float north  = tIn[N];
    float below  = temp1;          // replicated boundary
    float above  = temp3;

    // FMA-friendly evaluation
    float acc = cc * center;
    acc = fmaf(cw, west,  acc);
    acc = fmaf(ce, east,  acc);
    acc = fmaf(cs, south, acc);
    acc = fmaf(cn, north, acc);
    acc = fmaf(cb, below, acc);
    acc = fmaf(ct, above, acc);
    acc = fmaf(sdc, p[c], acc);
    acc = fmaf(ct, amb_temp, acc);
    tOut[c] = acc;

    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;

    // Middle layers: 1 .. nz-2
    for (int k = 1; k < nz - 1; ++k) {
        // Roll z-values
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c + xy];

        center = temp2;
        west   = tIn[W];
        east   = tIn[E];
        south  = tIn[S];
        north  = tIn[N];
        below  = temp1;
        above  = temp3;

        acc = cc * center;
        acc = fmaf(cw, west,  acc);
        acc = fmaf(ce, east,  acc);
        acc = fmaf(cs, south, acc);
        acc = fmaf(cn, north, acc);
        acc = fmaf(cb, below, acc);
        acc = fmaf(ct, above, acc);
        acc = fmaf(sdc, p[c], acc);
        acc = fmaf(ct, amb_temp, acc);
        tOut[c] = acc;

        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }

    // Last layer (k = nz-1)
    temp1 = temp2;
    temp2 = temp3;

    center = temp2;
    west   = tIn[W];
    east   = tIn[E];
    south  = tIn[S];
    north  = tIn[N];
    below  = temp1;
    above  = temp3;  // replicated boundary

    acc = cc * center;
    acc = fmaf(cw, west,  acc);
    acc = fmaf(ce, east,  acc);
    acc = fmaf(cs, south, acc);
    acc = fmaf(cn, north, acc);
    acc = fmaf(cb, below, acc);
    acc = fmaf(ct, above, acc);
    acc = fmaf(sdc, p[c], acc);
    acc = fmaf(ct, amb_temp, acc);
    tOut[c] = acc;
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
