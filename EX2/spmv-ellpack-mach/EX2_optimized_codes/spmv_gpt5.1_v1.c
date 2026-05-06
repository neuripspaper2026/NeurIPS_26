#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../spmv.h"

static double spmv_ellpack_kernel_time_acc = 0.0;

void reset_spmv_ellpack_kernel_time(void) { spmv_ellpack_kernel_time_acc = 0.0; }
double get_spmv_ellpack_kernel_time(void) { return spmv_ellpack_kernel_time_acc; }

void ellpack(TYPE nzval[N*L], int32_t cols[N*L], TYPE vec[N], TYPE out[N])
{
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    // Outer loop parallelization with OpenMP; schedule can be tuned via env vars if needed
    #ifdef _OPENMP
    #pragma omp parallel for default(none) shared(nzval, cols, vec, out) schedule(static)
    #endif
    for (int i = 0; i < N; i++) {
        // Initialize sum from current out[i] as in original code
        TYPE sum = out[i];
        int base = i * L;

        // Fully unrolled inner loop for fixed small L = 10
        // This improves ILP and removes loop control overhead
        sum += nzval[base + 0] * vec[cols[base + 0]];
        sum += nzval[base + 1] * vec[cols[base + 1]];
        sum += nzval[base + 2] * vec[cols[base + 2]];
        sum += nzval[base + 3] * vec[cols[base + 3]];
        sum += nzval[base + 4] * vec[cols[base + 4]];
        sum += nzval[base + 5] * vec[cols[base + 5]];
        sum += nzval[base + 6] * vec[cols[base + 6]];
        sum += nzval[base + 7] * vec[cols[base + 7]];
        sum += nzval[base + 8] * vec[cols[base + 8]];
        sum += nzval[base + 9] * vec[cols[base + 9]];

        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_ellpack_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
