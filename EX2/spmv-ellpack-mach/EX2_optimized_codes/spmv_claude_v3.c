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
        TYPE s0, s1, s2, s3;
        int idx;
        
        s0 = s1 = s2 = s3 = 0.0;
        
        ellpack_2 : for (j=0; j<L-3; j+=4) {
            idx = j + i*L;
            s0 += nzval[idx] * vec[cols[idx]];
            s1 += nzval[idx+1] * vec[cols[idx+1]];
            s2 += nzval[idx+2] * vec[cols[idx+2]];
            s3 += nzval[idx+3] * vec[cols[idx+3]];
        }
        
        sum += s0 + s1 + s2 + s3;
        
        for (; j<L; j++) {
            idx = j + i*L;
            sum += nzval[idx] * vec[cols[idx]];
        }
        
        out[i] = sum;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    spmv_ellpack_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                    (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
