#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../gemm.h"

static double gemm_ncubed_kernel_time_acc = 0.0;

void reset_gemm_ncubed_kernel_time(void) { gemm_ncubed_kernel_time_acc = 0.0; }
double get_gemm_ncubed_kernel_time(void) { return gemm_ncubed_kernel_time_acc; }

void gemm(TYPE m1[N], TYPE m2[N], TYPE prod[N]) {
  int i, j, k;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  /* Precompute the transpose of m2 to improve memory locality:
     m2 is stored row-major as m2[k * col_size + j]
     m2T is stored row-major such that m2T[j * row_size + k] = m2[k * col_size + j] */
  TYPE m2T[N];

  /* Parallelize the transpose; purely write-after-read, safe to parallelize */
#ifdef _OPENMP
#pragma omp parallel for private(i, j) schedule(static)
#endif
  for (i = 0; i < row_size; ++i) {
    int i_col = i * col_size;
    for (j = 0; j < col_size; ++j) {
      m2T[j * row_size + i] = m2[i_col + j];
    }
  }

  /* GEMM computation: prod = m1 * m2
     Now accesses:
       m1[i * col_size + k]  contiguous in k
       m2T[j * row_size + k] contiguous in k
     which gives unit-stride accesses for both matrices in the innermost loop. */
#ifdef _OPENMP
#pragma omp parallel for private(i, j, k) schedule(static)
#endif
  for (i = 0; i < row_size; ++i) {
    int i_col = i * col_size;
    for (j = 0; j < col_size; ++j) {
      TYPE sum = 0.0;
      int j_row = j * row_size;
      for (k = 0; k < row_size; ++k) {
        sum += m1[i_col + k] * m2T[j_row + k];
      }
      prod[i_col + j] = sum;
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  gemm_ncubed_kernel_time_acc +=
      (kernel_end.tv_sec - kernel_start.tv_sec) +
      (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
