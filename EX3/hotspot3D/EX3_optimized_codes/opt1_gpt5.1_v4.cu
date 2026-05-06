long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdio.h>

long long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000000) + tv.tv_usec;
}

__global__ void hotspotOpt1(float * __restrict__ p,
                            float * __restrict__ tIn,
                            float * __restrict__ tOut,
                            float sdc,
                            int nx, int ny, int nz,
                            float ce, float cw,
                            float cn, float cs,
                            float ct, float cb,
                            float cc) {
    const float amb_temp = 80.0f;

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int c  = i + j * nx;
    int xy = nx * ny;

    int W = (i == 0)      ? c : c - 1;
    int E = (i == nx - 1) ? c : c + 1;
    int N = (j == 0)      ? c : c - nx;
    int S = (j == ny - 1) ? c : c + nx;

    float temp1, temp2, temp3;

    // k = 0 layer
    temp1 = temp2 = __ldg(&tIn[c]);
    temp3 = (nz > 1) ? __ldg(&tIn[c + xy]) : temp2;

    float tW = __ldg(&tIn[W]);
    float tE = __ldg(&tIn[E]);
    float tN = __ldg(&tIn[N]);
    float tS = __ldg(&tIn[S]);
    float p0 = __ldg(&p[c]);

    tOut[c] = fmaf(cc, temp2,
              fmaf(cw, tW,
              fmaf(ce, tE,
              fmaf(cs, tS,
              fmaf(cn, tN,
              fmaf(cb, temp1,
              fmaf(ct, temp3,
              fmaf(sdc, p0, ct * amb_temp))))))));

    c += xy;
    W += xy;
    E += xy;
    N += xy;
    S += xy;

    // middle layers: 1 .. nz-2
    for (int k = 1; k < nz - 1; ++k) {
        temp1 = temp2;
        temp2 = temp3;
        temp3 = __ldg(&tIn[c + xy]);

        tW = __ldg(&tIn[W]);
        tE = __ldg(&tIn[E]);
        tN = __ldg(&tIn[N]);
        tS = __ldg(&tIn[S]);
        float pk = __ldg(&p[c]);

        tOut[c] = fmaf(cc, temp2,
                  fmaf(cw, tW,
                  fmaf(ce, tE,
                  fmaf(cs, tS,
                  fmaf(cn, tN,
                  fmaf(cb, temp1,
                  fmaf(ct, temp3,
                  fmaf(sdc, pk, ct * amb_temp))))))));

        c += xy;
        W += xy;
        E += xy;
        N += xy;
        S += xy;
    }

    // k = nz-1 layer (only if nz > 1)
    if (nz > 1) {
        temp1 = temp2;
        temp2 = temp3;

        tW = __ldg(&tIn[W]);
        tE = __ldg(&tIn[E]);
        tN = __ldg(&tIn[N]);
        tS = __ldg(&tIn[S]);
        float p_last = __ldg(&p[c]);

        tOut[c] = fmaf(cc, temp2,
                  fmaf(cw, tW,
                  fmaf(ce, tE,
                  fmaf(cs, tS,
                  fmaf(cn, tN,
                  fmaf(cb, temp1,
                  fmaf(ct, temp3,
                  fmaf(sdc, p_last, ct * amb_temp))))))));
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

    cc = 1.0f - (2.0f * ce + 2.0f * cn + 3.0f * ct);

    size_t s = sizeof(float) * (size_t)nx * (size_t)ny * (size_t)nz;
    float *tIn_d, *tOut_d, *p_d;
    cudaMalloc((void **)&p_d, s);
    cudaMalloc((void **)&tIn_d, s);
    cudaMalloc((void **)&tOut_d, s);
    cudaMemcpy(tIn_d, tIn, s, cudaMemcpyHostToDevice);
    cudaMemcpy(p_d, p, s, cudaMemcpyHostToDevice);

    cudaFuncSetCacheConfig(hotspotOpt1, cudaFuncCachePreferL1);

    dim3 block_dim(128, 2, 1);
    dim3 grid_dim((nx + block_dim.x - 1) / block_dim.x,
                  (ny + block_dim.y - 1) / block_dim.y,
                  1);

    long long start = get_time();
    for (int i = 0; i < numiter; ++i) {
        hotspotOpt1<<<grid_dim, block_dim>>>(p_d, tIn_d, tOut_d, stepDivCap, nx,
                                             ny, nz, ce, cw, cn, cs, ct, cb,
                                             cc);
        float *t = tIn_d;
        tIn_d = tOut_d;
        tOut_d = t;
    }
    cudaDeviceSynchronize();
    long long stop = get_time();
    float time = (float)((stop - start) / (1000.0 * 1000.0));
    printf("Time: %.3f (s)\n", time);
    cudaMemcpy(tOut, tOut_d, s, cudaMemcpyDeviceToHost);
    cudaFree(p_d);
    cudaFree(tIn_d);
    cudaFree(tOut_d);
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
