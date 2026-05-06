long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

__global__ void hotspotOpt1(float *p, float *tIn, float *tOut, float sdc,
                            int nx, int ny, int nz, float ce, float cw,
                            float cn, float cs, float ct, float cb, float cc) {
    // Use constant for ambient temperature to allow better compiler optimization
    const float amb_temp = 80.0f;

    // Compute global indices
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;

    // Guard against out-of-range threads (for non-multiple grid dims)
    if (i >= nx || j >= ny) return;

    // Precompute frequently used values
    const int xy = nx * ny;
    const int c_base = i + j * nx;

    // Precompute clamped neighbor indices in x/y once
    const int W = (i == 0)       ? c_base : c_base - 1;
    const int E = (i == nx - 1)  ? c_base : c_base + 1;
    const int N = (j == 0)       ? c_base : c_base - nx;
    const int S = (j == ny - 1)  ? c_base : c_base + nx;

    // Register prefetching for z-dimension traversal
    int c  = c_base;
    int Wc = W;
    int Ec = E;
    int Nc = N;
    int Sc = S;

    // Load center and neighbor in +z direction
    float temp1, temp2, temp3;
    temp1 = temp2 = __ldg(&tIn[c]);
    temp3 = __ldg(&tIn[c + xy]);

    // Precompute constant term from ambient temperature
    const float ct_amb = ct * amb_temp;

    // First slice (k = 0)
    tOut[c] = (cc * temp2) +
              (cw * __ldg(&tIn[Wc])) +
              (ce * __ldg(&tIn[Ec])) +
              (cs * __ldg(&tIn[Sc])) +
              (cn * __ldg(&tIn[Nc])) +
              (cb * temp1) +
              (ct * temp3) +
              (sdc * __ldg(&p[c])) +
              ct_amb;

    // Advance indices to next z-slice
    c  += xy;
    Wc += xy;
    Ec += xy;
    Nc += xy;
    Sc += xy;

#pragma unroll 4
    for (int k = 1; k < nz - 1; ++k) {
        // Rotate registers for z-neighbors
        temp1 = temp2;
        temp2 = temp3;

        // Prefetch next +z neighbor
        temp3 = __ldg(&tIn[c + xy]);

        // Stencil computation for interior slices
        tOut[c] = (cc * temp2) +
                  (cw * __ldg(&tIn[Wc])) +
                  (ce * __ldg(&tIn[Ec])) +
                  (cs * __ldg(&tIn[Sc])) +
                  (cn * __ldg(&tIn[Nc])) +
                  (cb * temp1) +
                  (ct * temp3) +
                  (sdc * __ldg(&p[c])) +
                  ct_amb;

        // Move to next slice
        c  += xy;
        Wc += xy;
        Ec += xy;
        Nc += xy;
        Sc += xy;
    }

    // Last slice (k = nz - 1), reuse last available z+1 value in temp3
    temp1 = temp2;
    temp2 = temp3;

    tOut[c] = (cc * temp2) +
              (cw * __ldg(&tIn[Wc])) +
              (ce * __ldg(&tIn[Ec])) +
              (cs * __ldg(&tIn[Sc])) +
              (cn * __ldg(&tIn[Nc])) +
              (cb * temp1) +
              (ct * temp3) +
              (sdc * __ldg(&p[c])) +
              ct_amb;

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
