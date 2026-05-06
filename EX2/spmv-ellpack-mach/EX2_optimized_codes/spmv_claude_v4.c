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
        TYPE Si0, Si1, Si2, Si3, Si4;
        int idx0, idx1, idx2, idx3, idx4;
        int base = i*L;
        
        ellpack_2 : for (j=0; j<L-4; j+=5) {
            idx0 = base + j;
            idx1 = base + j + 1;
            idx2 = base + j + 2;
            idx3 = base + j + 3;
            idx4 = base + j + 4;
            
            Si0 = nzval[idx0] * vec[cols[idx0]];
            Si1 = nzval[idx1] * vec[cols[idx1]];
            Si2 = nzval[idx2] * vec[cols[idx2]];
            Si3 = nzval[idx3] * vec[cols[idx3]];
            Si4 = nzval[idx4] * vec[cols[idx4]];
            
            sum += Si0 + Si1 + Si2 + Si3 + Si4;
        }
        
        for (; j<L; j++) {
            idx0 = base + j;
            Si0 = nzval[idx0] * vec[cols[idx0]];
            sum += Si0;
        }
        
        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_ellpack_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
