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

#ifdef _OPENMP
    #pragma omp parallel for private(j) schedule(static)
#endif
    ellpack_1 : for (i=0; i<N; i++) {
        TYPE sum = out[i];
        TYPE Si0, Si1, Si2, Si3;
        int idx0, idx1, idx2, idx3;
        int col0, col1, col2, col3;
        
        ellpack_2 : for (j=0; j<L-3; j+=4) {
            idx0 = j + i*L;
            idx1 = idx0 + 1;
            idx2 = idx0 + 2;
            idx3 = idx0 + 3;
            
            col0 = cols[idx0];
            col1 = cols[idx1];
            col2 = cols[idx2];
            col3 = cols[idx3];
            
            Si0 = nzval[idx0] * vec[col0];
            Si1 = nzval[idx1] * vec[col1];
            Si2 = nzval[idx2] * vec[col2];
            Si3 = nzval[idx3] * vec[col3];
            
            sum += Si0 + Si1 + Si2 + Si3;
        }
        
        for (; j<L; j++) {
            int idx = j + i*L;
            sum += nzval[idx] * vec[cols[idx]];
        }
        
        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_ellpack_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
