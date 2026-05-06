#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../gemm.h"

static double gemm_ncubed_kernel_time_acc = 0.0;

void reset_gemm_ncubed_kernel_time(void) { gemm_ncubed_kernel_time_acc = 0.0; }
double get_gemm_ncubed_kernel_time(void) { return gemm_ncubed_kernel_time_acc; }

void gemm( TYPE m1[N], TYPE m2[N], TYPE prod[N] ){
    int i, j, k;
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    /* Block sizes tuned for small 64x64 problem; can be adjusted */
    const int BI = 8;
    const int BJ = 8;
    const int BK = 8;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) private(i,j,k) schedule(static)
#endif
    for (int ii = 0; ii < row_size; ii += BI) {
        for (int jj = 0; jj < col_size; jj += BJ) {

            int iimax = (ii + BI < row_size) ? (ii + BI) : row_size;
            int jjmax = (jj + BJ < col_size) ? (jj + BJ) : col_size;

            for (int kk = 0; kk < row_size; kk += BK) {
                int kkmax = (kk + BK < row_size) ? (kk + BK) : row_size;

                for (i = ii; i < iimax; i++) {
                    int i_col = i * col_size;
                    for (j = jj; j < jjmax; j++) {
                        TYPE sum = (kk == 0) ? 0.0 : prod[i_col + j];
                        for (k = kk; k < kkmax; k++) {
                            sum += m1[i_col + k] * m2[k * col_size + j];
                        }
                        prod[i_col + j] = sum;
                    }
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
