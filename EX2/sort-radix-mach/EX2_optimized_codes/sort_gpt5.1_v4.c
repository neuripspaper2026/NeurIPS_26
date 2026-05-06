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

// Assuming sort_radix_kernel_time_acc is defined elsewhere as in original environment
extern double sort_radix_kernel_time_acc;

void ss_sort(int a[SIZE], int b[SIZE], int bucket[BUCKETSIZE], int sum[SCAN_RADIX]){
    int exp = 0;
    int valid_buffer = 0;
    #define BUFFER_A 0
    #define BUFFER_B 1
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    for (exp = 0; exp < 32; exp += 2) {
        int use_a_for_hist = (valid_buffer == BUFFER_A);

        init(bucket);

        /* Histogram: parallelize over blocks, each thread writes to distinct indices */
        if (use_a_for_hist) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
                int base = blockID * ELEMENTSPERBLOCK;
                int off0 = ((a[base] >> exp) & MASK) * NUMOFBLOCKS + blockID + 1;
                int off1 = ((a[base + 1] >> exp) & MASK) * NUMOFBLOCKS + blockID + 1;
                int off2 = ((a[base + 2] >> exp) & MASK) * NUMOFBLOCKS + blockID + 1;
                int off3 = ((a[base + 3] >> exp) & MASK) * NUMOFBLOCKS + blockID + 1;
                bucket[off0]++;
                bucket[off1]++;
                bucket[off2]++;
                bucket[off3]++;
            }
        } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
                int base = blockID * ELEMENTSPERBLOCK;
                int off0 = ((b[base] >> exp) & MASK) * NUMOFBLOCKS + blockID + 1;
                int off1 = ((b[base + 1] >> exp) & MASK) * NUMOFBLOCKS + blockID + 1;
                int off2 = ((b[base + 2] >> exp) & MASK) * NUMOFBLOCKS + blockID + 1;
                int off3 = ((b[base + 3] >> exp) & MASK) * NUMOFBLOCKS + blockID + 1;
                bucket[off0]++;
                bucket[off1]++;
                bucket[off2]++;
                bucket[off3]++;
            }
        }

        /* Local scan: each SCAN_BLOCK-sized stripe can be scanned independently */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int radixID = 0; radixID < SCAN_RADIX; radixID++) {
            int base = radixID * SCAN_BLOCK;
            for (int i = 1; i < SCAN_BLOCK; i++) {
                bucket[base + i] += bucket[base + i - 1];
            }
        }

        /* Sum scan: serial dependency over stripes */
        sum[0] = 0;
        for (int radixID = 1; radixID < SCAN_RADIX; radixID++) {
            int bucket_indx = radixID * SCAN_BLOCK - 1;
            sum[radixID] = sum[radixID - 1] + bucket[bucket_indx];
        }

        /* Last step scan: add per-stripe offsets; stripes are independent */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int radixID = 0; radixID < SCAN_RADIX; radixID++) {
            int base = radixID * SCAN_BLOCK;
            int add = sum[radixID];
            for (int i = 0; i < SCAN_BLOCK; i++) {
                bucket[base + i] += add;
            }
        }

        /* Update: parallel over blocks, each element writes to unique index */
        if (valid_buffer == BUFFER_A) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
                int base = blockID * ELEMENTSPERBLOCK;

                int a0 = a[base];
                int a1 = a[base + 1];
                int a2 = a[base + 2];
                int a3 = a[base + 3];

                int idx0 = ((a0 >> exp) & MASK) * NUMOFBLOCKS + blockID;
                int idx1 = ((a1 >> exp) & MASK) * NUMOFBLOCKS + blockID;
                int idx2 = ((a2 >> exp) & MASK) * NUMOFBLOCKS + blockID;
                int idx3 = ((a3 >> exp) & MASK) * NUMOFBLOCKS + blockID;

                int pos0 = bucket[idx0];
                bucket[idx0] = pos0 + 1;
                int pos1 = bucket[idx1];
                bucket[idx1] = pos1 + 1;
                int pos2 = bucket[idx2];
                bucket[idx2] = pos2 + 1;
                int pos3 = bucket[idx3];
                bucket[idx3] = pos3 + 1;

                b[pos0] = a0;
                b[pos1] = a1;
                b[pos2] = a2;
                b[pos3] = a3;
            }
            valid_buffer = BUFFER_B;
        } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
                int base = blockID * ELEMENTSPERBLOCK;

                int b0 = b[base];
                int b1 = b[base + 1];
                int b2 = b[base + 2];
                int b3 = b[base + 3];

                int idx0 = ((b0 >> exp) & MASK) * NUMOFBLOCKS + blockID;
                int idx1 = ((b1 >> exp) & MASK) * NUMOFBLOCKS + blockID;
                int idx2 = ((b2 >> exp) & MASK) * NUMOFBLOCKS + blockID;
                int idx3 = ((b3 >> exp) & MASK) * NUMOFBLOCKS + blockID;

                int pos0 = bucket[idx0];
                bucket[idx0] = pos0 + 1;
                int pos1 = bucket[idx1];
                bucket[idx1] = pos1 + 1;
                int pos2 = bucket[idx2];
                bucket[idx2] = pos2 + 1;
                int pos3 = bucket[idx3];
                bucket[idx3] = pos3 + 1;

                a[pos0] = b0;
                a[pos1] = b1;
                a[pos2] = b2;
                a[pos3] = b3;
            }
            valid_buffer = BUFFER_A;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    sort_radix_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
