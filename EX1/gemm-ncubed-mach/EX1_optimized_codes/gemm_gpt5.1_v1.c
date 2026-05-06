#include <time.h>
#include "../gemm.h"

static double gemm_ncubed_kernel_time_acc = 0.0;

void reset_gemm_ncubed_kernel_time(void) { gemm_ncubed_kernel_time_acc = 0.0; }
double get_gemm_ncubed_kernel_time(void) { return gemm_ncubed_kernel_time_acc; }

void gemm( TYPE m1[N], TYPE m2[N], TYPE prod[N] ){
    struct timespec kernel_start, kernel_end;
    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (int i = 0; i < row_size; i++) {
        int i_col = i * col_size;
        for (int j = 0; j < col_size; j++) {
            TYPE sum = 0.0;

            int k = 0;
            int j0 = j;
            int j1 = j0 + col_size;
            int j2 = j1 + col_size;
            int j3 = j2 + col_size;

            /* Unroll inner loop by factor of 4 */
            for (; k + 3 < row_size; k += 4) {
                int k0 = k;
                int k1 = k0 + 1;
                int k2 = k0 + 2;
                int k3 = k0 + 3;

                TYPE a0 = m1[i_col + k0];
                TYPE a1 = m1[i_col + k1];
                TYPE a2 = m1[i_col + k2];
                TYPE a3 = m1[i_col + k3];

                sum += a0 * m2[j0];
                sum += a1 * m2[j1];
                sum += a2 * m2[j2];
                sum += a3 * m2[j3];

                j0 += col_size * 4;
                j1 += col_size * 4;
                j2 += col_size * 4;
                j3 += col_size * 4;
            }

            /* Remainder loop (if row_size not multiple of 4) */
            for (; k < row_size; k++) {
                sum += m1[i_col + k] * m2[k * col_size + j];
            }

            prod[i_col + j] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    gemm_ncubed_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                   (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
