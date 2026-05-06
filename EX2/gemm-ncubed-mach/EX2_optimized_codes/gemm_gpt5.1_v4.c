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

    /* Block sizes tuned for small 64x64 problem; safe for general use */
    const int BI = 8;
    const int BJ = 8;
    const int BK = 8;

#ifdef _OPENMP
    /* Parallelize over outermost i-blocks; each thread works on private tiles */
#pragma omp parallel for private(j,k) schedule(static)
#endif
    for (i = 0; i < row_size; i += BI) {
        int i_end = i + BI;
        if (i_end > row_size) i_end = row_size;
        for (j = 0; j < col_size; j += BJ) {
            int j_end = j + BJ;
            if (j_end > col_size) j_end = col_size;

            /* Initialize C tile */
            for (int ii = i; ii < i_end; ++ii) {
                int i_col = ii * col_size;
                for (int jj = j; jj < j_end; ++jj) {
                    prod[i_col + jj] = 0.0;
                }
            }

            /* Compute C(i..i_end-1, j..j_end-1) tile */
            for (k = 0; k < row_size; k += BK) {
                int k_end = k + BK;
                if (k_end > row_size) k_end = row_size;

                for (int ii = i; ii < i_end; ++ii) {
                    int i_col = ii * col_size;
                    for (int kk = k; kk < k_end; ++kk) {
                        int k_col = kk * col_size;
                        TYPE a_val = m1[i_col + kk];
                        TYPE *restrict c_ptr = &prod[i_col + j];
                        TYPE *restrict b_ptr = &m2[k_col + j];
                        for (int jj = j; jj < j_end; ++jj) {
                            c_ptr[jj - j] += a_val * b_ptr[jj - j];
                        }
                    }
                }
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
