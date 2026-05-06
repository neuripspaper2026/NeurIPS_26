#include <time.h>
#include "../spmv.h"

static double spmv_ellpack_kernel_time_acc = 0.0;

void reset_spmv_ellpack_kernel_time(void) { spmv_ellpack_kernel_time_acc = 0.0; }
double get_spmv_ellpack_kernel_time(void) { return spmv_ellpack_kernel_time_acc; }

void ellpack(TYPE nzval[N*L], int32_t cols[N*L], TYPE vec[N], TYPE out[N])
{
    int i, j;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    ellpack_1 : for (i=0; i<N; i++) {
        TYPE sum = out[i];
        TYPE s0, s1, s2, s3;
        int idx0, idx1, idx2, idx3;
        
        ellpack_2 : for (j=0; j<L-3; j+=4) {
            idx0 = j + i*L;
            idx1 = idx0 + 1;
            idx2 = idx0 + 2;
            idx3 = idx0 + 3;
            
            s0 = nzval[idx0] * vec[cols[idx0]];
            s1 = nzval[idx1] * vec[cols[idx1]];
            s2 = nzval[idx2] * vec[cols[idx2]];
            s3 = nzval[idx3] * vec[cols[idx3]];
            
            sum += s0 + s1 + s2 + s3;
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
