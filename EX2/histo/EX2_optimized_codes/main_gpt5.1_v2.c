#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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
  int early_error = 0; /* track errors to avoid early returns */

  if (!parameters)
    early_error = -1;

  int numIterations = 0;
  unsigned int img_width = 0, img_height = 0;
  unsigned int histo_width = 0, histo_height = 0;
  unsigned int *img = NULL;
  unsigned char *histo = NULL;
  FILE *f = NULL;
  int result = 0;

  if (!early_error) {
    if (!parameters->inpFiles[0]) {
      fputs("Input file expected\n", stderr);
      early_error = -1;
    }
  }

  if (!early_error) {
    if (argc >= 2) {
      numIterations = atoi(argv[1]);
      if (numIterations <= 0) {
        fputs("Iteration count must be positive\n", stderr);
        early_error = -1;
      }
    } else {
      fputs("Expected at least one command line argument\n", stderr);
      early_error = -1;
    }
  }

  pb_InitializeTimerSet(&timers);

  char *inputStr = (char *)"Input";
  char *outputStr = (char *)"Output";

  pb_AddSubTimer(&timers, inputStr, pb_TimerID_IO);
  pb_AddSubTimer(&timers, outputStr, pb_TimerID_IO);

  pb_SwitchToSubTimer(&timers, inputStr, pb_TimerID_IO);

  if (!early_error) {
    f = fopen(parameters->inpFiles[0],"rb");
    if (!f) {
      fputs("Error opening input file\n", stderr);
      early_error = -1;
    }
  }

  if (!early_error) {
    result  = (int)fread(&img_width,    sizeof(unsigned int), 1, f);
    result += (int)fread(&img_height,   sizeof(unsigned int), 1, f);
    result += (int)fread(&histo_width,  sizeof(unsigned int), 1, f);
    result += (int)fread(&histo_height, sizeof(unsigned int), 1, f);

    if (result != 4) {
      fputs("Error reading input and output dimensions from file\n", stderr);
      early_error = -1;
    }
  }

  if (!early_error) {
    img = (unsigned int*) malloc((size_t)img_width * (size_t)img_height * sizeof(unsigned int));
    histo = (unsigned char*) calloc((size_t)histo_width * (size_t)histo_height, sizeof(unsigned char));
    if (!img || !histo) {
      fputs("Error allocating memory\n", stderr);
      early_error = -1;
    }
  }

  pb_SwitchToSubTimer(&timers, inputStr, pb_TimerID_IO);

  if (!early_error) {
    const size_t npix = (size_t)img_width * (size_t)img_height;
    result = (int)fread(img, sizeof(unsigned int), npix, f);

    if (f) {
      fclose(f);
      f = NULL;
    }

    if ((size_t)result != npix) {
      fputs("Error reading input array from file\n", stderr);
      early_error = -1;
    }
  } else {
    if (f) {
      fclose(f);
      f = NULL;
    }
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  int iter;
  /* Start timing for kernel execution */
  struct timespec kernel_start, kernel_end;
  clock_gettime(CLOCK_MONOTONIC, &kernel_start);

  if (!early_error) {
    const size_t histo_size = (size_t)histo_height * (size_t)histo_width;
    const size_t npix = (size_t)img_width * (size_t)img_height;

    for (iter = 0; iter < numIterations; iter++) {
      /* Zero histogram */
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (size_t i = 0; i < histo_size; ++i) {
        histo[i] = 0;
      }

      /* Parallel histogram using per-thread private histograms */
#ifdef _OPENMP
      int max_val = (int)histo_size;
#pragma omp parallel
      {
        int nth = omp_get_num_threads();
        int tid = omp_get_thread_num();
        size_t chunk = (histo_size + (size_t)nth - 1) / (size_t)nth;
        size_t start = (size_t)tid * chunk;
        if (start > histo_size) start = histo_size;
        size_t end = start + chunk;
        if (end > histo_size) end = histo_size;
        size_t local_size = end - start;

        unsigned char *local_histo = NULL;
        if (local_size > 0) {
          local_histo = (unsigned char*)calloc(local_size, sizeof(unsigned char));
        }

#pragma omp for schedule(static)
        for (size_t i = 0; i < npix; ++i) {
          unsigned int value = img[i];
          if ((int)value < max_val) {
            if (value >= start && value < end) {
              size_t idx = (size_t)value - start;
              if (local_histo[idx] < UINT8_MAX) {
                local_histo[idx]++;
              }
            }
          }
        }

        if (local_histo) {
#pragma omp critical
          {
            for (size_t i = 0; i < local_size; ++i) {
              unsigned int sum = (unsigned int)histo[start + i] + (unsigned int)local_histo[i];
              if (sum > UINT8_MAX) sum = UINT8_MAX;
              histo[start + i] = (unsigned char)sum;
            }
          }
          free(local_histo);
        }
      }
#else
      /* Serial histogram */
      for (size_t i = 0; i < npix; ++i) {
        const unsigned int value = img[i];
        if (value < histo_size && histo[value] < UINT8_MAX) {
          ++histo[value];
        }
      }
#endif
    }
  }

  /* End timing for kernel execution */
  clock_gettime(CLOCK_MONOTONIC, &kernel_end);
  double kernel_time = (kernel_end.tv_sec - kernel_start.tv_sec) +
                       (kernel_end.tv_nsec - kernel_start.tv_nsec) / 1e9;

//  pb_SwitchToTimer(&timers, pb_TimerID_IO);
  pb_SwitchToSubTimer(&timers, outputStr, pb_TimerID_IO);

  if (!early_error) {
    if (parameters->outFile) {
      dump_histo_img(histo, histo_height, histo_width, parameters->outFile);
    }
  }

  pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

  if (img) free(img);
  if (histo) free(histo);

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);

  printf("\n");
  pb_PrintTimerSet(&timers);
  if (parameters)
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

  return early_error;
}
