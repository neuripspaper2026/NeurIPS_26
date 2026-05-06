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

void local_scan(int bucket[BUCKETSIZE])
{
    int radixID, i, bucket_indx;
    for (radixID = 0; radixID < SCAN_RADIX; radixID++) {
        /* Each SCAN_BLOCK is independent; vectorize the inner scan. */
#pragma omp simd
        for (i = 1; i < SCAN_BLOCK; i++) {
            bucket_indx = radixID * SCAN_BLOCK + i;
            bucket[bucket_indx] += bucket[bucket_indx - 1];
        }
    }
}

void sum_scan(int sum[SCAN_RADIX], int bucket[BUCKETSIZE])
{
    int radixID, bucket_indx;
    sum[0] = 0;
    for (radixID = 1; radixID < SCAN_RADIX; radixID++) {
        bucket_indx = radixID * SCAN_BLOCK - 1;
        sum[radixID] = sum[radixID - 1] + bucket[bucket_indx];
    }
}

void last_step_scan(int bucket[BUCKETSIZE], int sum[SCAN_RADIX])
{
    int radixID, i, bucket_indx;
    for (radixID = 0; radixID < SCAN_RADIX; radixID++) {
        /* All bucket indices within a radixID are independent; parallelize. */
#pragma omp parallel for private(i, bucket_indx) default(none) shared(bucket, sum, radixID)
        for (i = 0; i < SCAN_BLOCK; i++) {
            bucket_indx = radixID * SCAN_BLOCK + i;
            bucket[bucket_indx] = bucket[bucket_indx] + sum[radixID];
        }
    }
}

void init(int bucket[BUCKETSIZE])
{
    int i;
    /* Initialize bucket in parallel; each element written once. */
#pragma omp parallel for private(i) default(none) shared(bucket)
    for (i = 0; i < BUCKETSIZE; i++) {
        bucket[i] = 0;
    }
}

void hist(int bucket[BUCKETSIZE], int a[SIZE], int exp)
{
    int blockID, i, bucket_indx, a_indx;

    /* Parallelize over blocks; use per-thread private histogram to avoid races,
       then reduce into the shared bucket. */
#pragma omp parallel default(none) shared(bucket, a, exp) private(blockID, i, bucket_indx, a_indx)
    {
        int tid = 0;
        int nthreads = 1;
#ifdef _OPENMP
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
#endif
        int blocks_per_thread = (NUMOFBLOCKS + nthreads - 1) / nthreads;
        int start_block = tid * blocks_per_thread;
        int end_block = start_block + blocks_per_thread;
        if (end_block > NUMOFBLOCKS) end_block = NUMOFBLOCKS;

        /* Private histogram for this thread */
        int local_bucket[BUCKETSIZE];
        for (int bi = 0; bi < BUCKETSIZE; bi++) {
            local_bucket[bi] = 0;
        }

        for (blockID = start_block; blockID < end_block; blockID++) {
#pragma omp simd private(i, a_indx, bucket_indx)
            for (i = 0; i < 4; i++) {
                a_indx = blockID * ELEMENTSPERBLOCK + i;
                bucket_indx = ((a[a_indx] >> exp) & 0x3) * NUMOFBLOCKS + blockID + 1;
                local_bucket[bucket_indx]++;
            }
        }

        /* Reduce local_bucket into global bucket. */
#pragma omp for nowait
        for (int bi = 0; bi < BUCKETSIZE; bi++) {
            /* Each bi is updated by multiple threads; use atomic to avoid races. */
#pragma omp atomic
            bucket[bi] += local_bucket[bi];
        }
    }
}

void update(int b[SIZE], int bucket[BUCKETSIZE], int a[SIZE], int exp)
{
    int blockID, i, bucket_indx, a_indx;

    /* Parallelize over blocks; each block writes to disjoint regions of b,
       but bucket increments are not independent, so we update via atomics. */
#pragma omp parallel for private(blockID, i, bucket_indx, a_indx) default(none) shared(b, bucket, a, exp)
    for (blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
#pragma omp simd
        for (i = 0; i < 4; i++) {
            int idx = blockID * ELEMENTSPERBLOCK + i;
            bucket_indx = ((a[idx] >> exp) & 0x3) * NUMOFBLOCKS + blockID;
            a_indx = idx;
            int pos;
#pragma omp atomic capture
            { pos = bucket[bucket_indx]; bucket[bucket_indx]++; }
            b[pos] = a[a_indx];
        }
    }
}

void ss_sort(int a[SIZE], int b[SIZE], int bucket[BUCKETSIZE], int sum[SCAN_RADIX]){
    int exp=0;
    int valid_buffer=0;
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
