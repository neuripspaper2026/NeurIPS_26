#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <assert.h>
#include <time.h>

#define WRITE_OUTPUT
#define CHECK_OUTPUT

#ifndef OUTPUT_FILE_NAME
#define OUTPUT_FILE_NAME "output.data"
#endif

#include "support.h"

#ifdef BENCHMARK_TIMING
extern void BENCHMARK_RESET_FUNC(void);
extern double BENCHMARK_GET_FUNC(void);
#endif

int main(int argc, char **argv)
{
  struct timespec main_start, main_end;
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &main_start);

  FILE *timing_file = stderr;
  const char *timing_path = getenv("TIMING_LOG_FILE");
  if (timing_path && timing_path[0] != '\0') {
    FILE *tmp = fopen(timing_path, "w");
    if (tmp) {
      timing_file = tmp;
    } else {
      fprintf(stderr, "[WARN] Failed to open TIMING_LOG_FILE=%s for writing, falling back to stderr.\n", timing_path);
    }
  }

  // Parse command line.
  char *in_file;
  #ifdef CHECK_OUTPUT
  char *check_file;
  #endif
  assert( argc<4 && "Usage: ./benchmark <input_file> <check_file>" );
  in_file = "input.data";
  #ifdef CHECK_OUTPUT
  check_file = "check.data";
  #endif
  if( argc>1 )
    in_file = argv[1];
  #ifdef CHECK_OUTPUT
  if( argc>2 )
    check_file = argv[2];
  #endif

  // Load input data
  int in_fd;
  char *data;
  data = malloc(INPUT_SIZE);
  assert( data!=NULL && "Out of memory" );
  in_fd = open( in_file, O_RDONLY );
  assert( in_fd>0 && "Couldn't open input data file");
  input_to_data(in_fd, data);
  
  // Unpack and call
#ifdef BENCHMARK_TIMING
  BENCHMARK_RESET_FUNC();
#endif
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);
  run_benchmark( data );
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);

  #ifdef WRITE_OUTPUT
  int out_fd;
  out_fd = open(OUTPUT_FILE_NAME, O_WRONLY|O_CREAT|O_TRUNC, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
  assert( out_fd>0 && "Couldn't open output data file" );
  data_to_output(out_fd, data);
  close(out_fd);
  #endif

  // Load check data
  #ifdef CHECK_OUTPUT
  int check_fd;
  char *ref;
  ref = malloc(INPUT_SIZE);
  assert( ref!=NULL && "Out of memory" );
  check_fd = open( check_file, O_RDONLY );
  assert( check_fd>0 && "Couldn't open check data file");
  output_to_data(check_fd, ref);
  #endif

  // Validate benchmark results
  #ifdef CHECK_OUTPUT
  if( !check_data(data, ref) ) {
    fprintf(stderr, "Benchmark results are incorrect\n");
    return -1;
  }
  #endif
  free(data);
  free(ref);

  printf("Success.\n");
  clock_gettime(CLOCK_MONOTONIC, &main_end);

#ifdef BENCHMARK_TIMING
  double kernel_time = BENCHMARK_GET_FUNC();
  if (kernel_time == 0.0) {
    kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                  (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
  }
#else
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;
#endif
  double main_time = (main_end.tv_sec - main_start.tv_sec) +
                     (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

  fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
  fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);
  fflush(timing_file);
  if (timing_file != stderr) {
    fclose(timing_file);
  }

  return 0;
}
