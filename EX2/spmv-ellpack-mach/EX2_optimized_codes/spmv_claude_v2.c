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

    #pragma omp parallel for schedule(static) private(j)
    ellpack_1 : for (i=0; i<N; i++) {
        TYPE sum = out[i];
        TYPE Si0, Si1, Si2, Si3, Si4, Si5, Si6, Si7, Si8, Si9;
        int idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7, idx8, idx9;
        int base = i*L;
        
        // Manual unroll by 10 (L=10)
        idx0 = base + 0;
        idx1 = base + 1;
        idx2 = base + 2;
        idx3 = base + 3;
        idx4 = base + 4;
        idx5 = base + 5;
        idx6 = base + 6;
        idx7 = base + 7;
        idx8 = base + 8;
        idx9 = base + 9;
        
        Si0 = nzval[idx0] * vec[cols[idx0]];
        Si1 = nzval[idx1] * vec[cols[idx1]];
        Si2 = nzval[idx2] * vec[cols[idx2]];
        Si3 = nzval[idx3] * vec[cols[idx3]];
        Si4 = nzval[idx4] * vec[cols[idx4]];
        Si5 = nzval[idx5] * vec[cols[idx5]];
        Si6 = nzval[idx6] * vec[cols[idx6]];
        Si7 = nzval[idx7] * vec[cols[idx7]];
        Si8 = nzval[idx8] * vec[cols[idx8]];
        Si9 = nzval[idx9] * vec[cols[idx9]];
        
        sum += Si0 + Si1 + Si2 + Si3 + Si4 + Si5 + Si6 + Si7 + Si8 + Si9;
        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_ellpack_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
