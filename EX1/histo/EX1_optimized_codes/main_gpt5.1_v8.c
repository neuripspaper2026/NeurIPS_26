#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "util.h"

#define UINT8_MAX 255

int main(int argc, char* argv[]) {
  /* Start timing for total execution */
  struct timespec main_start, main_end;
  clock_gettime(CLOCK_MONOTONIC, &main_start);

  struct pb_TimerSet timers;
  struct pb_Parameters *parameters;

  printf("Base implementation of histogramming.\n");
  printf("Maintained by Nady Obeid <obeid1@ece.uiuc.edu>\n");

  parameters = pb_ReadParameters(&argc, argv);
  if (!parameters)
    return -1;

  if (!parameters->inpFiles[0]) {
    fputs("Input file expected\n", stderr);
    return -1;
  }

  int numIterations;
  if (argc >= 2) {
    numIterations = atoi(argv[1]);
  } else {
    fputs("Expected at least one command line argument\n", stderr);
    return -1;
  }

  pb_InitializeTimerSet(&timers);

  char *inputStr = "Input";
  char *outputStr = "Output";

  pb_AddSubTimer(&timers, inputStr, pb_TimerID_IO);
  pb_AddSubTimer(&timers, outputStr, pb_TimerID_IO);

  pb_SwitchToSubTimer(&timers, inputStr, pb_TimerID_IO);

  unsigned int img_width, img_height;
  unsigned int histo_width, histo_height;

  FILE* f = fopen(parameters->inpFiles[0], "rb");
  if (!f) {
    fputs("Error opening input file\n", stderr);
    return -1;
  }

  size_t read_count = 0;
  read_count += fread(&img_width,    sizeof(unsigned int), 1, f);
  read_count += fread(&img_height,   sizeof(unsigned int), 1, f);
  read_count += fread(&histo_width,  sizeof(unsigned int), 1, f);
  read_count += fread(&histo_height, sizeof(unsigned int), 1, f);

  if (read_count != 4) {
    fputs("Error reading input and output dimensions from file\n", stderr);
    fclose(f);
    return -1;
  }

  const size_t img_nelts   = (size_t)img_width * (size_t)img_height;
  const size_t histo_nelts = (size_t)histo_width * (size_t)histo_height;

  unsigned int* img = (unsigned int*)malloc(img_nelts * sizeof(unsigned int));
  unsigned char* histo = (unsigned char*)calloc(histo_nelts, sizeof(unsigned char));

  if (!img || !histo) {
    fputs("Memory allocation failure\n", stderr);
    free(img);
    free(histo);
    fclose(f);
    return -1;
  }

  pb_SwitchToSubTimer(&timers, "Input", pb_TimerID_IO);

  read_count = fread(img, sizeof(unsigned int), img_nelts, f);

  fclose(f);

  if (read_count != img_nelts) {
    fputs("Error reading input array from file\n", stderr);
    free(img);
    free(histo);
    return -1;
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  int iter;
  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  for (iter = 0; iter < numIterations; iter++) {
    memset(histo, 0, histo_nelts * sizeof(unsigned char));

    const unsigned int *restrict img_ptr = img;
    const unsigned int *restrict img_end = img + img_nelts;
    unsigned char *restrict histo_ptr = histo;
    const unsigned int histo_limit = (unsigned int)histo_nelts;

    while (img_ptr != img_end) {
      const unsigned int value = *img_ptr++;
      if (value < histo_limit) {
        if (histo_ptr[value] < UINT8_MAX) {
          ++histo_ptr[value];
        }
      }
    }
  }

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

  pb_SwitchToSubTimer(&timers, outputStr, pb_TimerID_IO);

  if (parameters->outFile) {
    dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  free(img);
  free(histo);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  printf("\n");
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(parameters);

  /* End timing for total execution */
  clock_gettime(CLOCK_MONOTONIC, &main_end);
  double main_time = (main_end.tv_sec - main_start.tv_sec) +
                     (main_end.tv_nsec - main_start.tv_nsec) / 1e9;

  /* Determine timing output destination */
  FILE *timing_file = stderr;
  const char *timing_path = getenv("TIMING_LOG_FILE");
  if (timing_path && timing_path[0] != '\0') {
    FILE *tmp = fopen(timing_path, "w");
    if (tmp)
      timing_file = tmp;
  }

  /* Print timing results */
  fprintf(timing_file, "KERNEL_TIME: %.9f\n", kernel_time);
  fprintf(timing_file, "TOTAL_TIME: %.9f\n", main_time);

  if (timing_file != stderr)
    fclose(timing_file);

  return 0;
}
