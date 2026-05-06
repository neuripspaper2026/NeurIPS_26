long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

__global__ void hotspotOpt1(float *p, float *tIn, float *tOut, float sdc,
                            int nx, int ny, int nz, float ce, float cw,
                            float cn, float cs, float ct, float cb, float cc) {
    // Use constant for ambient temperature to keep it in registers
    const float amb_temp = 80.0f;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int c = i + j * nx;
    int xy = nx * ny;

    int W = (i == 0) ? c : c - 1;
    int E = (i == nx - 1) ? c : c + 1;
    int N = (j == 0) ? c : c - nx;
    int S = (j == ny - 1) ? c : c + nx;

    // Preload neighbors for z = 0 plane
    float temp1, temp2, temp3;
    temp1 = temp2 = __ldg(&tIn[c]);
    temp3 = __ldg(&tIn[c + xy]);

    float tW = __ldg(&tIn[W]);
    float tE = __ldg(&tIn[E]);
    float tN = __ldg(&tIn[N]);
    float tS = __ldg(&tIn[S]);
    float pVal = __ldg(&p[c]);

    // First layer (k = 0)
    tOut[c] = cc * temp2 + cw * tW + ce * tE + cs * tS +
              cn * tN + cb * temp1 + ct * temp3 + sdc * pVal +
              ct * amb_temp;

    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;

    // Iterate through middle layers (1 .. nz-2)
    for (int k = 1; k < nz - 1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = __ldg(&tIn[c + xy]);

        tW = __ldg(&tIn[W]);
        tE = __ldg(&tIn[E]);
        tN = __ldg(&tIn[N]);
        tS = __ldg(&tIn[S]);
        pVal = __ldg(&p[c]);

        tOut[c] = cc * temp2 + cw * tW + ce * tE + cs * tS +
                  cn * tN + cb * temp1 + ct * temp3 + sdc * pVal +
                  ct * amb_temp;

        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }

    // Last layer (k = nz-1)
    temp1 = temp2;
    temp2 = temp3;

    tW = __ldg(&tIn[W]);
    tE = __ldg(&tIn[E]);
    tN = __ldg(&tIn[N]);
    tS = __ldg(&tIn[S]);
    pVal = __ldg(&p[c]);

    tOut[c] = cc * temp2 + cw * tW + ce * tE + cs * tS +
              cn * tN + cb * temp1 + ct * temp3 + sdc * pVal +
              ct * amb_temp;
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
