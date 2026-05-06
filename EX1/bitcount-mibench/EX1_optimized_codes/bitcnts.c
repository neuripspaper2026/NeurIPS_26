#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <time.h>
#include <float.h>
#include "../bitops.h"

#define FUNCS  7

static int CDECL bit_shifter(long int x);

static double bit_shifter_kernel_time_acc = 0.0;

void reset_bit_shifter_kernel_time(void) { bit_shifter_kernel_time_acc = 0.0; }
double get_bit_shifter_kernel_time(void) { return bit_shifter_kernel_time_acc; }

int main(int argc, char *argv[])
{
  int ret = 0;
  struct timespec main_start, main_end;
  clock_gettime(CLOCK_MONOTONIC, &main_start);
  FILE *timing_file = stderr;
  const char *timing_path = getenv("TIMING_LOG_FILE");
  if (timing_path && timing_path[0] != '\0') {
    FILE *tmp = fopen(timing_path, "w");
    if (tmp) {
      timing_file = tmp;
    }
  }
  reset_bit_shifter_kernel_time();

  clock_t start, stop;
  double ct, cmin = DBL_MAX, cmax = 0;
  int i, cminix = 0, cmaxix = 0;
  long j, n, seed;
  int iterations;
  int bits_only = 0; /* if true, only print 7 lines of Bits totals */
  static int (* CDECL pBitCntFunc[FUNCS])(long) = {
    bit_count,
    bitcount,
    ntbl_bitcnt,
    ntbl_bitcount,
    /*            btbl_bitcnt, DOESNT WORK*/
    BW_btbl_bitcount,
    AR_btbl_bitcount,
    bit_shifter
  };
  static char *text[FUNCS] = {
    "Optimized 1 bit/loop counter",
    "Ratko's mystery algorithm",
    "Recursive bit count by nybbles",
    "Non-recursive bit count by nybbles",
    /*            "Recursive bit count by bytes",*/
    "Non-recursive bit count by bytes (BW)",
    "Non-recursive bit count by bytes (AR)",
    "Shift and count bits"
  };
  if (argc<2) {
    fprintf(stderr,"Usage: bitcnts <iterations> [--bits-only]\n");
    ret = -1;
    goto timing_cleanup;
  }
  iterations=atoi(argv[1]);
  if (argc >= 3) {
    bits_only = (!strcmp(argv[2], "--bits-only") || !strcmp(argv[2], "--bits") || !strcmp(argv[2], "-b"));
  }

  /* Fix random seed for reproducibility */
  srand(12345);
  
  if (!bits_only) {
    puts("Bit counter algorithm benchmark\n");
  }
  
  for (i = 0; i < FUNCS; i++) {
    start = clock();
    
    for (j = n = 0, seed = rand(); j < iterations; j++, seed += 13)
	 n += pBitCntFunc[i](seed);
    
    stop = clock();
    ct = (stop - start) / (double)CLOCKS_PER_SEC;
    if (ct < cmin) {
	 cmin = ct;
	 cminix = i;
    }
    if (ct > cmax) {
	 cmax = ct;
	 cmaxix = i;
    }
    
    if (bits_only) {
      /* Only print the Bits total (one line per algorithm) */
      printf("%ld\n", n);
    } else {
      printf("%-38s> Time: %7.3f sec.; Bits: %ld\n", text[i], ct, n);
    }
  }
  if (!bits_only) {
    printf("\nBest  > %s\n", text[cminix]);
    printf("Worst > %s\n", text[cmaxix]);
  }
  ret = 0;

timing_cleanup:
  clock_gettime(CLOCK_MONOTONIC, &main_end);
  double kernel_time = get_bit_shifter_kernel_time();
  double main_time = (main_end.tv_sec - main_start.tv_sec) +
                     (main_end.tv_nsec - main_start.tv_nsec) / 1e9;
  fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
  fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);
  fflush(timing_file);
  if (timing_file != stderr) {
    fclose(timing_file);
  }
  return ret;
}

static int CDECL bit_shifter(long int x)
{
  int i, n;
  struct timespec kernel_start, kernel_end;

  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  for (i = n = 0; x && (i < (int)(sizeof(long) * CHAR_BIT)); ++i, x >>= 1)
    n += (int)(x & 1L);

  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  bit_shifter_kernel_time_acc += (kernel_end.tv_sec - kernel_start.tv_sec) +
                                 (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
  return n;
}
