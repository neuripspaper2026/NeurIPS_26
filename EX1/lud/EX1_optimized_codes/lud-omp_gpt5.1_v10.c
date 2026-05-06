#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    printf("num of threads = %d\n", omp_num_threads);

    for (i = 0; i < size; ++i) {
        const int row_offset_i = i * size;

        /* Upper (U) part: a[i, j], j = i..size-1 */
        for (j = i; j < size; ++j) {
            float sum = a[row_offset_i + j];
            float *row_i_ptr = a + row_offset_i;   /* points to a[i,0] */
            float *col_j_ptr = a + j;              /* points to a[0,j] */

            for (k = 0; k < i; ++k) {
                sum -= row_i_ptr[k] * col_j_ptr[k * size];
            }
            a[row_offset_i + j] = sum;
        }

        /* Lower (L) part: a[j, i], j = i+1..size-1 */
        const float diag = a[row_offset_i + i];
        for (j = i + 1; j < size; ++j) {
            const int row_offset_j = j * size;
            float sum = a[row_offset_j + i];
            float *row_j_ptr = a + row_offset_j;   /* points to a[j,0] */
            float *col_i_ptr = a + i;              /* points to a[0,i] */

            for (k = 0; k < i; ++k) {
                sum -= row_j_ptr[k] * col_i_ptr[k * size];
            }
            a[row_offset_j + i] = sum / diag;
        }
    }
}
