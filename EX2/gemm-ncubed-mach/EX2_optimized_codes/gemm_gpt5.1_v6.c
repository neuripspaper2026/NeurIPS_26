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

    /* Loop ordering and blocking for better cache usage, plus OpenMP parallelization */
    const int MR = 8;   /* block size for i */
    const int NR = 8;   /* block size for j */

    int ii, jj;
#ifdef _OPENMP
#pragma omp parallel for private(ii,jj,i,j,k) schedule(static)
#endif
    for (ii = 0; ii < row_size; ii += MR) {
        for (jj = 0; jj < col_size; jj += NR) {
            int i_end = (ii + MR < row_size) ? (ii + MR) : row_size;
            int j_end = (jj + NR < col_size) ? (jj + NR) : col_size;

            for (i = ii; i < i_end; ++i) {
                int i_col = i * col_size;
                for (j = jj; j < j_end; ++j) {
                    TYPE sum = 0.0;

                    /* Inner loop over k; keep indexing simple for compiler vectorization */
                    for (k = 0; k < row_size; ++k) {
                        int k_col = k * col_size;
                        sum += m1[i_col + k] * m2[k_col + j];
                    }
                    prod[i_col + j] = sum;
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
