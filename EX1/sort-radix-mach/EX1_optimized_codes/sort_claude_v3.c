#include <time.h>
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

void ss_sort(int a[SIZE], int b[SIZE], int bucket[BUCKETSIZE], int sum[SCAN_RADIX]){
    int exp=0;
    int valid_buffer=0;
    #define BUFFER_A 0
    #define BUFFER_B 1
    struct timespec kernel_start, kernel_end;

    clock_gettime(CLOCK_MONOTONIC, &kernel_start);

    sort_1 : for (exp=0; exp<32; exp+=2) {
        // Inline init
        for (int i=0; i<BUCKETSIZE; i++) {
            bucket[i] = 0;
        }

        // Inline hist
        int *src = (valid_buffer == BUFFER_A) ? a : b;
        for (int blockID=0; blockID<NUMOFBLOCKS; blockID++) {
            int base_idx = blockID * ELEMENTSPERBLOCK;
            for(int i=0; i<4; i++) {
                int bucket_indx = ((src[base_idx + i] >> exp) & 0x3)*NUMOFBLOCKS + blockID + 1;
                bucket[bucket_indx]++;
            }
        }

        // Inline local_scan
        for (int radixID=0; radixID<SCAN_RADIX; radixID++) {
            int base = radixID*SCAN_BLOCK;
            for (int i=1; i<SCAN_BLOCK; i++){
                bucket[base + i] += bucket[base + i - 1];
            }
        }

        // Inline sum_scan
        sum[0] = 0;
        for (int radixID=1; radixID<SCAN_RADIX; radixID++) {
            sum[radixID] = sum[radixID-1] + bucket[radixID*SCAN_BLOCK - 1];
        }

        // Inline last_step_scan
        for (int radixID=0; radixID<SCAN_RADIX; radixID++) {
            int base = radixID * SCAN_BLOCK;
            int sum_val = sum[radixID];
            for (int i=0; i<SCAN_BLOCK; i++) {
                bucket[base + i] += sum_val;
            }
        }

        // Inline update
        int *dst = (valid_buffer == BUFFER_A) ? b : a;
        for (int blockID = 0; blockID < NUMOFBLOCKS; blockID++) {
            int base_idx = blockID * ELEMENTSPERBLOCK;
            for(int i=0; i<4; i++) {
                int a_val = src[base_idx + i];
                int bucket_indx = ((a_val >> exp) & 0x3)*NUMOFBLOCKS + blockID;
                dst[bucket[bucket_indx]] = a_val;
                bucket[bucket_indx]++;
            }
        }
        
        valid_buffer = (valid_buffer == BUFFER_A) ? BUFFER_B : BUFFER_A;
    }

    clock_gettime(CLOCK_MONOTONIC, &kernel_end);
    sort_radix_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
}
