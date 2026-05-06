#include <stdio.h>

extern int omp_num_threads;

void lud_omp(float *a, int size) {
    int i, j, k;
    float sum;
    float *pi, *pk, *pj;
    printf("num of threads = %d\n", omp_num_threads);
    for (i = 0; i < size; i++) {
        pi = a + i * size;
        for (j = i; j < size; j++) {
            sum = pi[j];
            pk = a;
            for (k = 0; k < i; k++, pk += size)
                sum -= pi[k] * pk[j];
            pi[j] = sum;
        }
        for (j = i + 1; j < size; j++) {
            sum = a[j * size + i];
            pk = a;
            for (k = 0; k < i; k++, pk += size)
                sum -= a[j * size + k] * pk[i];
            a[j * size + i] = sum / pi[i];
        }
    }
}
