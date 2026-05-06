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
    TYPE Si;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    ellpack_1 : for (i=0; i<N; i++) {
        TYPE sum = out[i];
        ellpack_2 : for (j=0; j<L; j++) {
                Si = nzval[j + i*L] * vec[cols[j + i*L]];
                sum += Si;
        }
        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_ellpack_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
