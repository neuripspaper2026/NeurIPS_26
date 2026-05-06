#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../../common_MachSuite/support.h"

#define SIZE 2048
#define TYPE int32_t
#define TYPE_MAX INT32_MAX

void ms_mergesort(TYPE a[SIZE]);
void reset_sort_merge_kernel_time(void);
double get_sort_merge_kernel_time(void);

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  TYPE a[SIZE];
};
