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
    const int c0 = i + j * nx;

    const bool is_west  = (i == 0);
    const bool is_east  = (i == nx - 1);
    const bool is_north = (j == 0);
    const bool is_south = (j == ny - 1);

    int W = is_west  ? c0 : c0 - 1;
    int E = is_east  ? c0 : c0 + 1;
    int N = is_north ? c0 : c0 - nx;
    int S = is_south ? c0 : c0 + nx;

    int c = c0;

    // Preload first three z-planes to reduce redundant global loads
    float temp1 = tIn[c];           // T(z-1) for interior loop, reused
    float temp2 = temp1;            // T(z)
    float temp3 = tIn[c + xy];      // T(z+1)

    // First slice (k = 0)
    tOut[c] = cc * temp2 +
              cw * tIn[W] + ce * tIn[E] +
              cs * tIn[S] + cn * tIn[N] +
              cb * temp1 + ct * temp3 +
              sdc * p[c] + ct * amb_temp;

    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;

    // Main z-loop, unrolled by 2 for better ILP on A100
    int k = 1;
    const int nzm1 = nz - 1;

#pragma unroll 2
    for (; k < nzm1 - 1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c + xy];

        float center = temp2;
        float west   = tIn[W];
        float east   = tIn[E];
        float south  = tIn[S];
        float north  = tIn[N];
        float below  = temp1;
        float above  = temp3;
        float power  = p[c];

        tOut[c] = cc * center +
                  cw * west + ce * east +
                  cs * south + cn * north +
                  cb * below + ct * above +
                  sdc * power + ct * amb_temp;

        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }

    // Handle last interior slice (k = nz-2) and last slice (k = nz-1)

    // k = nz-2 (only if nz > 2)
    if (nzm1 > 1) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = tIn[c + xy];

        float center = temp2;
        float west   = tIn[W];
        float east   = tIn[E];
        float south  = tIn[S];
        float north  = tIn[N];
        float below  = temp1;
        float above  = temp3;
        float power  = p[c];

        tOut[c] = cc * center +
                  cw * west + ce * east +
                  cs * south + cn * north +
                  cb * below + ct * above +
                  sdc * power + ct * amb_temp;

        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;

        temp1 = temp2;
        temp2 = temp3;
    } else {
        // nz == 2, we have only first and last slice
        temp1 = temp2;
        temp2 = temp3;
    }

    // Last slice (k = nz-1), reuse temp3 as "above" to keep arithmetic consistent
    {
        float center = temp2;
        float west   = tIn[W];
        float east   = tIn[E];
        float south  = tIn[S];
        float north  = tIn[N];
        float below  = temp1;
        float above  = temp3;
        float power  = p[c];

        tOut[c] = cc * center +
                  cw * west + ce * east +
                  cs * south + cn * north +
                  cb * below + ct * above +
                  sdc * power + ct * amb_temp;
    }
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
