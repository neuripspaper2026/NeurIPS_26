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

extern double sort_radix_kernel_time_acc;

void local_scan(int bucket[BUCKETSIZE])
{
    int radixID, i, bucket_indx;
    for (radixID = 0; radixID < SCAN_RADIX; radixID++) {
        int base = radixID * SCAN_BLOCK;
        for (i = 1; i < SCAN_BLOCK; i++) {
            bucket_indx = base + i;
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
        int base_sum = sum[radixID];
        int base_idx = radixID * SCAN_BLOCK;
        for (i = 0; i < SCAN_BLOCK; i++) {
            bucket_indx = base_idx + i;
            bucket[bucket_indx] += base_sum;
        }
    }
}

void init(int bucket[BUCKETSIZE])
{
    int i;
    int n = BUCKETSIZE;
    for (i = 0; i < n; i++) {
        bucket[i] = 0;
    }
}

void hist(int bucket[BUCKETSIZE], int a[SIZE], int exp)
{
    int blockID, i, bucket_indx, a_indx;
    const int numBlocks = NUMOFBLOCKS;
    const int elementsPerBlock = ELEMENTSPERBLOCK;
    const int mask = MASK;
    const int numBlocksLocal = NUMOFBLOCKS;

    for (blockID = 0; blockID < numBlocks; blockID++) {
        int base_a = blockID * elementsPerBlock;
        for (i = 0; i < elementsPerBlock; i++) {
            a_indx = base_a + i;
            int key = (a[a_indx] >> exp) & mask;
            bucket_indx = key * numBlocksLocal + blockID + 1;
            bucket[bucket_indx]++;
        }
    }
}

void update(int b[SIZE], int bucket[BUCKETSIZE], int a[SIZE], int exp)
{
    int i, blockID, bucket_indx, a_indx;
    const int numBlocks = NUMOFBLOCKS;
    const int elementsPerBlock = ELEMENTSPERBLOCK;
    const int mask = MASK;
    const int numBlocksLocal = NUMOFBLOCKS;

    for (blockID = 0; blockID < numBlocks; blockID++) {
        int base_a = blockID * elementsPerBlock;
        for (i = 0; i < elementsPerBlock; i++) {
            a_indx = base_a + i;
            int key = (a[a_indx] >> exp) & mask;
            bucket_indx = key * numBlocksLocal + blockID;
            int pos = bucket[bucket_indx]++;
            b[pos] = a[a_indx];
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

        /* Parallel initialization of bucket */
        {
            int i;
            int n = BUCKETSIZE;
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (i = 0; i < n; i++) {
                bucket[i] = 0;
            }
        }

        if (valid_buffer == BUFFER_A) {
            /* Parallel histogram on current buffer a */
            #ifdef _OPENMP
            #pragma omp parallel
            {
                int blockID, i;
                int elementsPerBlock = ELEMENTSPERBLOCK;
                int mask = MASK;
                int numBlocksLocal = NUMOFBLOCKS;
                #pragma omp for schedule(static)
                for (blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
                    int base_a = blockID * elementsPerBlock;
                    for (i = 0; i < elementsPerBlock; i++) {
                        int a_indx = base_a + i;
                        int key = (a[a_indx] >> exp) & mask;
                        int bucket_indx = key * numBlocksLocal + blockID + 1;
                        #pragma omp atomic
                        bucket[bucket_indx]++;
                    }
                }
            }
            #else
            hist(bucket, a, exp);
            #endif
        } else {
            /* Parallel histogram on current buffer b */
            #ifdef _OPENMP
            #pragma omp parallel
            {
                int blockID, i;
                int elementsPerBlock = ELEMENTSPERBLOCK;
                int mask = MASK;
                int numBlocksLocal = NUMOFBLOCKS;
                #pragma omp for schedule(static)
                for (blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
                    int base_b = blockID * elementsPerBlock;
                    for (i = 0; i < elementsPerBlock; i++) {
                        int b_indx = base_b + i;
                        int key = (b[b_indx] >> exp) & mask;
                        int bucket_indx = key * numBlocksLocal + blockID + 1;
                        #pragma omp atomic
                        bucket[bucket_indx]++;
                    }
                }
            }
            #else
            hist(bucket, b, exp);
            #endif
        }

        /* Local scan: each block is small; keep serial per block but parallelize across blocks */
        {
            int radixID;
            #ifdef _OPENMP
            #pragma omp parallel for private(radixID) schedule(static)
            #endif
            for (radixID = 0; radixID < SCAN_RADIX; radixID++) {
                int i;
                int base = radixID * SCAN_BLOCK;
                for (i = 1; i < SCAN_BLOCK; i++) {
                    int idx = base + i;
                    bucket[idx] += bucket[idx - 1];
                }
            }
        }

        /* Serial scan over sums: dependency chain is long and short loop; keep serial */
        sum_scan(sum, bucket);

        /* Last step scan can be parallelized across radix blocks */
        {
            int radixID;
            #ifdef _OPENMP
            #pragma omp parallel for private(radixID) schedule(static)
            #endif
            for (radixID = 0; radixID < SCAN_RADIX; radixID++) {
                int i;
                int base_sum = sum[radixID];
                int base_idx = radixID * SCAN_BLOCK;
                for (i = 0; i < SCAN_BLOCK; i++) {
                    int idx = base_idx + i;
                    bucket[idx] += base_sum;
                }
            }
        }

        if (valid_buffer == BUFFER_A) {
            /* Parallel update from a to b */
            #ifdef _OPENMP
            #pragma omp parallel
            {
                int blockID, i;
                int elementsPerBlock = ELEMENTSPERBLOCK;
                int mask = MASK;
                int numBlocksLocal = NUMOFBLOCKS;
                #pragma omp for schedule(static)
                for (blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
                    int base_a = blockID * elementsPerBlock;
                    for (i = 0; i < elementsPerBlock; i++) {
                        int a_indx = base_a + i;
                        int key = (a[a_indx] >> exp) & mask;
                        int bucket_indx = key * numBlocksLocal + blockID;
                        int pos;
                        #pragma omp atomic capture
                        { pos = bucket[bucket_indx]++; }
                        b[pos] = a[a_indx];
                    }
                }
            }
            #else
            update(b, bucket, a, exp);
            #endif
            valid_buffer = BUFFER_B;
        } else {
            /* Parallel update from b to a */
            #ifdef _OPENMP
            #pragma omp parallel
            {
                int blockID, i;
                int elementsPerBlock = ELEMENTSPERBLOCK;
                int mask = MASK;
                int numBlocksLocal = NUMOFBLOCKS;
                #pragma omp for schedule(static)
                for (blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
                    int base_b = blockID * elementsPerBlock;
                    for (i = 0; i < elementsPerBlock; i++) {
                        int b_indx = base_b + i;
                        int key = (b[b_indx] >> exp) & mask;
                        int bucket_indx = key * numBlocksLocal + blockID;
                        int pos;
                        #pragma omp atomic capture
                        { pos = bucket[bucket_indx]++; }
                        a[pos] = b[b_indx];
                    }
                }
            }
            #else
            update(a, bucket, b, exp);
            #endif
            valid_buffer = BUFFER_A;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    sort_radix_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
