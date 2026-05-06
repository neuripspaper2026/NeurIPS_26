#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../sort.h"

static double sort_radix_kernel_time_acc = 0.0;

void reset_sort_radix_kernel_time(void) { sort_radix_kernel_time_acc = 0.0; }
double get_sort_radix_kernel_time(void) { return sort_radix_kernel_time_acc; }

void local_scan(int bucket[BUCKETSIZE])
{
    int radixID, i, bucket_indx;
    local_1 : for (radixID=0; radixID<SCAN_RADIX; radixID++) {
        local_2 : for (i=1; i<SCAN_BLOCK; i++){
            bucket_indx = radixID*SCAN_BLOCK + i;
            bucket[bucket_indx] += bucket[bucket_indx-1];
        }
    }
}

void sum_scan(int sum[SCAN_RADIX], int bucket[BUCKETSIZE])
{
    int radixID, bucket_indx;
    sum[0] = 0;
    sum_1 : for (radixID=1; radixID<SCAN_RADIX; radixID++) {
        bucket_indx = radixID*SCAN_BLOCK - 1;
        sum[radixID] = sum[radixID-1] + bucket[bucket_indx];
    }
}

void last_step_scan(int bucket[BUCKETSIZE], int sum[SCAN_RADIX])
{
    int radixID, i, bucket_indx;
    last_1:for (radixID=0; radixID<SCAN_RADIX; radixID++) {
        last_2:for (i=0; i<SCAN_BLOCK; i++) {
            bucket_indx = radixID * SCAN_BLOCK + i;
            bucket[bucket_indx] = bucket[bucket_indx] + sum[radixID];
         }
    }
}

void init(int bucket[BUCKETSIZE])
{
    int i;
    init_1 : for (i=0; i<BUCKETSIZE; i++) {
        bucket[i] = 0;
    }
}

void hist(int bucket[BUCKETSIZE], int a[SIZE], int exp)
{
    int blockID, i, bucket_indx, a_indx;
    blockID = 0;
    hist_1 : for (blockID=0; blockID<NUMOFBLOCKS; blockID++) {
        hist_2 : for(i=0; i<4; i++) {
            a_indx = blockID * ELEMENTSPERBLOCK + i;
            bucket_indx = ((a[a_indx] >> exp) & 0x3)*NUMOFBLOCKS + blockID + 1;
            bucket[bucket_indx]++;
        }
    }
}

void update(int b[SIZE], int bucket[BUCKETSIZE], int a[SIZE], int exp)
{
    int i, blockID, bucket_indx, a_indx;
    blockID = 0;

    update_1 : for (blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
        update_2 : for(i=0; i<4; i++) {
            bucket_indx = ((a[blockID * ELEMENTSPERBLOCK + i] >> exp) & 0x3)*NUMOFBLOCKS + blockID;
            a_indx = blockID * ELEMENTSPERBLOCK + i;
            b[bucket[bucket_indx]] = a[a_indx];
            bucket[bucket_indx]++;
        }
    }
}

#include "sort.h"

#ifdef _OPENMP
#include <omp.h>
#endif

static double sort_radix_kernel_time_acc = 0.0;

void reset_sort_radix_kernel_time(void) {
    sort_radix_kernel_time_acc = 0.0;
}

double get_sort_radix_kernel_time(void) {
    return sort_radix_kernel_time_acc;
}

/*
Implementation based on algorithm described in:
A. Danalis, G. Marin, C. McCurdy, J. S. Meredith, P. C. Roth, K. Spafford, V. Tipparaju, and J. S. Vetter.
The scalable heterogeneous computing (shoc) benchmark suite.
In Proceedings of the 3rd Workshop on General-Purpose Computation on Graphics Processing Units, 2010
*/

void local_scan(int bucket[BUCKETSIZE])
{
    int radixID, i, bucket_indx;
#ifdef _OPENMP
#pragma omp parallel for private(i, bucket_indx) schedule(static)
#endif
    for (radixID = 0; radixID < SCAN_RADIX; radixID++) {
        for (i = 1; i < SCAN_BLOCK; i++) {
            bucket_indx = radixID * SCAN_BLOCK + i;
            bucket[bucket_indx] += bucket[bucket_indx - 1];
        }
    }
}

void sum_scan(int sum[SCAN_RADIX], int bucket[BUCKETSIZE])
{
    int radixID, bucket_indx;

    /* First create an array of block sums from bucket to improve locality. */
    int block_sum[SCAN_RADIX];
#ifdef _OPENMP
#pragma omp parallel for private(bucket_indx) schedule(static)
#endif
    for (radixID = 0; radixID < SCAN_RADIX; radixID++) {
        bucket_indx = radixID * SCAN_BLOCK - 1;
        if (radixID == 0) {
            /* For radixID 0, bucket_indx = -1 would be invalid; treat sum as 0. */
            block_sum[radixID] = 0;
        } else {
            block_sum[radixID] = bucket[bucket_indx];
        }
    }

    /* Sequential prefix-sum over block_sum into sum; dependency chain prevents
       efficient parallelization and SCAN_RADIX is small. */
    sum[0] = 0;
    for (radixID = 1; radixID < SCAN_RADIX; radixID++) {
        sum[radixID] = sum[radixID - 1] + block_sum[radixID];
    }
}

void last_step_scan(int bucket[BUCKETSIZE], int sum[SCAN_RADIX])
{
    int radixID, i, bucket_indx;
#ifdef _OPENMP
#pragma omp parallel for private(i, bucket_indx) schedule(static)
#endif
    for (radixID = 0; radixID < SCAN_RADIX; radixID++) {
        int base = sum[radixID];
        int start = radixID * SCAN_BLOCK;
        for (i = 0; i < SCAN_BLOCK; i++) {
            bucket_indx = start + i;
            bucket[bucket_indx] = bucket[bucket_indx] + base;
        }
    }
}

void init(int bucket[BUCKETSIZE])
{
    int i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < BUCKETSIZE; i++) {
        bucket[i] = 0;
    }
}

void hist(int bucket[BUCKETSIZE], int a[SIZE], int exp)
{
    int blockID, i, bucket_indx, a_indx;

    /* Use per-thread private histograms to avoid contention, then reduce. */
#ifdef _OPENMP
    int num_threads, t;
#pragma omp parallel private(blockID, i, bucket_indx, a_indx, t)
    {
        int tid = omp_get_thread_num();
#pragma omp single
        {
            num_threads = omp_get_num_threads();
        }

        /* Each thread works on a slice of blocks and maintains a private buffer. */
        int local_bucket[BUCKETSIZE];
        int j;
        for (j = 0; j < BUCKETSIZE; j++) {
            local_bucket[j] = 0;
        }

#pragma omp for schedule(static)
        for (blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
            int base = blockID * ELEMENTSPERBLOCK;
            for (i = 0; i < ELEMENTSPERBLOCK; i++) {
                a_indx = base + i;
                bucket_indx = ((a[a_indx] >> exp) & MASK) * NUMOFBLOCKS + blockID + 1;
                local_bucket[bucket_indx]++;
            }
        }

        /* Reduction: sum private histograms into global bucket. */
#pragma omp for schedule(static)
        for (j = 0; j < BUCKETSIZE; j++) {
            int sum_local = 0;
            int k;
#pragma omp simd reduction(+:sum_local)
            for (k = 0; k < 1; k++) {
                sum_local += local_bucket[j];
            }
            bucket[j] += sum_local;
        }
    }
#else
    for (blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
        int base = blockID * ELEMENTSPERBLOCK;
        for (i = 0; i < ELEMENTSPERBLOCK; i++) {
            a_indx = base + i;
            bucket_indx = ((a[a_indx] >> exp) & MASK) * NUMOFBLOCKS + blockID + 1;
            bucket[bucket_indx]++;
        }
    }
#endif
}

void update(int b[SIZE], int bucket[BUCKETSIZE], int a[SIZE], int exp)
{
    int i, blockID, bucket_indx, a_indx;

    /* The scan has made each bucket region independent; parallelize over blocks. */
#ifdef _OPENMP
#pragma omp parallel for private(i, bucket_indx, a_indx) schedule(static)
#endif
    for (blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
        int base = blockID * ELEMENTSPERBLOCK;
        for (i = 0; i < ELEMENTSPERBLOCK; i++) {
            a_indx = base + i;
            bucket_indx = ((a[a_indx] >> exp) & MASK) * NUMOFBLOCKS + blockID;
            int pos = bucket[bucket_indx];
            b[pos] = a[a_indx];
            bucket[bucket_indx] = pos + 1;
        }
    }
}

void ss_sort(int a[SIZE], int b[SIZE], int bucket[BUCKETSIZE], int sum[SCAN_RADIX]){
    int exp = 0;
    int valid_buffer = 0;
#define BUFFER_A 0
#define BUFFER_B 1
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (exp = 0; exp < 32; exp += 2) {
        init(bucket);

        if (valid_buffer == BUFFER_A) {
            hist(bucket, a, exp);
        } else {
            hist(bucket, b, exp);
        }

        local_scan(bucket);
        sum_scan(sum, bucket);
        last_step_scan(bucket, sum);

        if (valid_buffer == BUFFER_A) {
            update(b, bucket, a, exp);
            valid_buffer = BUFFER_B;
        } else {
            update(a, bucket, b, exp);
            valid_buffer = BUFFER_A;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    sort_radix_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
