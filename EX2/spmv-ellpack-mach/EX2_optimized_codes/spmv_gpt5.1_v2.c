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
    int i, j;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Parallelize outer loop over rows.
       Each iteration writes to a distinct out[i], so it's safe. */
#ifdef _OPENMP
#pragma omp parallel for private(j) schedule(static)
#endif
    for (i = 0; i < N; i++) {
        TYPE sum = out[i];

        /* Unroll inner loop for better ILP and fewer loop overheads.
           L is a small compile-time constant (=10). */
        for (j = 0; j < L; j += 2) {
            int base = i * L + j;

            TYPE nz0 = nzval[base];
            int c0 = cols[base];
            TYPE v0 = vec[c0];

            TYPE nz1 = nzval[base + 1];
            int c1 = cols[base + 1];
            TYPE v1 = vec[c1];

            sum += nz0 * v0 + nz1 * v1;
        }

        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_ellpack_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
