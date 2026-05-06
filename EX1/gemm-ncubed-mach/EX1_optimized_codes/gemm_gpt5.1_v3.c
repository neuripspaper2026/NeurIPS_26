#include <time.h>
#include "../gemm.h"

static double gemm_ncubed_kernel_time_acc = 0.0;

void reset_gemm_ncubed_kernel_time(void) { gemm_ncubed_kernel_time_acc = 0.0; }
double get_gemm_ncubed_kernel_time(void) { return gemm_ncubed_kernel_time_acc; }

void gemm(TYPE m1[N], TYPE m2[N], TYPE prod[N]) {
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  for (int i = 0; i < row_size; i++) {
    int const i_col = i * col_size;
    TYPE *restrict const m1_row = &m1[i_col];
    TYPE *restrict const prod_row = &prod[i_col];

    for (int j = 0; j < col_size; j++) {
      TYPE sum = 0.0;

      for (int k = 0; k < row_size; k++) {
        int const k_col = k * col_size;
        sum += m1_row[k] * m2[k_col + j];
      }

      prod_row[j] = sum;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  gemm_ncubed_kernel_time_acc +=
      (kernel_end.tv_sec - kernel_start.tv_sec) +
      (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
