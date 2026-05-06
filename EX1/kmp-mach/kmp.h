/*
Implementation based on http://www-igm.univ-mlv.fr/~lecroq/string/node8.html
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include "../../common_MachSuite/support.h"

#define PATTERN_SIZE 4
#define STRING_SIZE (32411)

int kmp(char pattern[PATTERN_SIZE], char input[STRING_SIZE], int32_t kmpNext[PATTERN_SIZE], int32_t n_matches[1]);
void reset_kmp_kernel_time(void);
double get_kmp_kernel_time(void);
////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  char pattern[PATTERN_SIZE];
  char input[STRING_SIZE];
  int32_t kmpNext[PATTERN_SIZE];
  int32_t n_matches[1];
};
